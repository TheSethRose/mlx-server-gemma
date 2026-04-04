"""
MLX Server - OpenAI-compatible API server for Apple Silicon.
Supports Gemma 4 and other mlx-vlm models with TurboQuant KV cache.
"""

import os
import re
import json
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import Optional, List

from mlx_vlm import load, generate
from mlx_vlm.models.base import load_chat_template

# Model state
model = None
processor = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global model, processor
    print(f"Loading model: {MODEL_NAME}")
    model, processor = load(MODEL_NAME)

    # Inject chat template if missing
    tokenizer = getattr(processor, "tokenizer", processor)
    if not getattr(tokenizer, "chat_template", None):
        print("No chat template found, loading from bundled template...")
        template_path = os.path.join(os.path.dirname(__file__), "chat_template.json")
        if os.path.exists(template_path):
            import pathlib
            load_chat_template(tokenizer, pathlib.Path(template_path).parent)
            print(f"Chat template loaded from {template_path}")
        else:
            print("WARNING: No chat_template.json found. Chat completions may fail.")

    num_layers = len(model.layers) if hasattr(model, "layers") else 0
    print(f"Model loaded: {num_layers} layers")
    print("Server ready.")
    yield
    print("Shutting down server...")


app = FastAPI(title="MLX Server", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL_NAME = os.environ.get("MLX_MODEL", "mlx-community/gemma-4-26b-a4b-bf16")
THINK_PATTERNS = (
    r"<think>.*?</think>",
    r"<thinking>.*?</thinking>",
    r"<reasoning>.*?</reasoning>",
    r"<REASONING_SCRATCHPAD>.*?</REASONING_SCRATCHPAD>",
)
TURN_BOUNDARIES = (
    "\nUser:",
    "\nAssistant:",
    "\nSystem:",
    "User:",
    "Assistant:",
    "System:",
)


class Message(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[Message]
    max_tokens: Optional[int] = 512
    temperature: Optional[float] = 0.7
    stream: Optional[bool] = False


def extract_visible_content(text: str) -> tuple[str, str]:
    raw = (text or "").strip()
    reasoning_parts = []

    for pattern in THINK_PATTERNS:
        reasoning_parts.extend(
            match.group(0)
            for match in re.finditer(pattern, raw, flags=re.DOTALL | re.IGNORECASE)
        )

    visible = raw
    for pattern in THINK_PATTERNS:
        visible = re.sub(pattern, "", visible, flags=re.DOTALL | re.IGNORECASE)

    for boundary in TURN_BOUNDARIES:
        if boundary in visible:
            visible = visible.split(boundary, 1)[0]

    return visible.strip(), "\n".join(reasoning_parts).strip()


@app.get("/v1/models")
async def list_models():
    return {
        "object": "list",
        "data": [{
            "id": MODEL_NAME,
            "object": "model",
            "created": 1234567890,
            "owned_by": "local",
        }]
    }


@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    if model is None or processor is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    messages = [{"role": msg.role, "content": msg.content} for msg in request.messages]
    tokenizer = getattr(processor, "tokenizer", processor)

    try:
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    except ValueError:
        raise HTTPException(
            status_code=500,
            detail="Chat template not available. Ensure chat_template.json exists in the model directory.",
        )

    try:
        result = generate(
            model,
            processor,
            prompt=prompt,
            max_tokens=request.max_tokens or 512,
            temperature=request.temperature or 0.7,
            verbose=False,
        )

        response_text = result.text
        visible, reasoning = extract_visible_content(response_text)
        final_response = visible or reasoning or response_text

        if request.stream:
            created = int(time.time())

            async def event_stream():
                role_chunk = {
                    "id": "chatcmpl-local",
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": request.model,
                    "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}],
                }
                yield f"data: {json.dumps(role_chunk, ensure_ascii=False)}\n\n"

                if final_response:
                    content_chunk = {
                        "id": "chatcmpl-local",
                        "object": "chat.completion.chunk",
                        "created": created,
                        "model": request.model,
                        "choices": [{"index": 0, "delta": {"content": final_response}, "finish_reason": None}],
                    }
                    yield f"data: {json.dumps(content_chunk, ensure_ascii=False)}\n\n"

                finish_chunk = {
                    "id": "chatcmpl-local",
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": request.model,
                    "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                    "usage": {
                        "prompt_tokens": len(tokenizer.encode(prompt)),
                        "completion_tokens": len(tokenizer.encode(final_response)),
                        "total_tokens": len(tokenizer.encode(prompt)) + len(tokenizer.encode(final_response)),
                    },
                }
                yield f"data: {json.dumps(finish_chunk, ensure_ascii=False)}\n\n"
                yield "data: [DONE]\n\n"

            return StreamingResponse(event_stream(), media_type="text/event-stream")

        return {
            "id": "chatcmpl-local",
            "object": "chat.completion",
            "created": 1234567890,
            "model": request.model,
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": final_response,
                },
                "finish_reason": "stop",
            }],
            "usage": {
                "prompt_tokens": len(tokenizer.encode(prompt)),
                "completion_tokens": len(tokenizer.encode(final_response)),
                "total_tokens": len(tokenizer.encode(prompt)) + len(tokenizer.encode(final_response)),
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health():
    return {"status": "ok", "model": MODEL_NAME}


if __name__ == "__main__":
    import argparse
    import uvicorn

    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--model", type=str, default=MODEL_NAME)
    args = parser.parse_args()

    MODEL_NAME = args.model

    print(f"Starting MLX server on {args.host}:{args.port}")
    print(f"Model: {MODEL_NAME}")

    uvicorn.run(app, host=args.host, port=args.port)
