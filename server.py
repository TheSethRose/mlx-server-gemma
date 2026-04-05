"""
MLX Server - OpenAI-compatible API server for Apple Silicon.
Supports Gemma 4 and other mlx-vlm models with TurboQuant KV cache.
"""

import os
import re
import json
import time
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import Optional, List, Dict, Any

from mlx_vlm import load, generate
from mlx_vlm.models.base import load_chat_template

model = None
processor = None
tokenizer = None

TOOL_CALL_PATTERN = re.compile(
    r"<\|tool_call\>call:(\w+)\{(.*?)\}<tool_call\|>",
    re.DOTALL,
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    global model, processor, tokenizer
    print(f"Loading model: {MODEL_NAME}")
    model, processor = load(MODEL_NAME)
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

MODEL_NAME = os.environ.get("MLX_MODEL", "mlx-community/gemma-4-26b-a4b-it-bf16")
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


class FunctionDef(BaseModel):
    name: str
    description: str
    parameters: Dict[str, Any]


class ToolDef(BaseModel):
    type: str = "function"
    function: FunctionDef


class Message(BaseModel):
    role: str
    content: Optional[str] = None
    tool_calls: Optional[List[Dict[str, Any]]] = None
    tool_call_id: Optional[str] = None


class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[Message]
    tools: Optional[List[ToolDef]] = None
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


def parse_tool_calls(text: str) -> list[dict]:
    matches = list(TOOL_CALL_PATTERN.finditer(text))
    if not matches:
        return []

    tool_calls = []
    for m in matches:
        name = m.group(1)
        args_str = m.group(2).strip()
        cleaned = args_str.replace('<|"|>', '"')
        cleaned = re.sub(r'(\w+):', r'"\1":', cleaned)
        try:
            args = json.loads("{" + cleaned + "}")
        except json.JSONDecodeError:
            args = {"_raw": args_str}
        tool_calls.append({
            "id": f"call_{uuid.uuid4().hex[:8]}",
            "type": "function",
            "function": {"name": name, "arguments": json.dumps(args)},
        })
    return tool_calls


def build_messages_for_template(messages: list) -> list:
    result = []
    for msg in messages:
        entry = {"role": msg.role}
        if msg.content is not None:
            entry["content"] = msg.content
        if msg.tool_calls:
            entry["tool_calls"] = msg.tool_calls
        if msg.tool_call_id:
            entry["tool_call_id"] = msg.tool_call_id
        result.append(entry)
    return result


def build_tool_declarations(tools: list) -> list:
    return [
        {
            "type": t.type,
            "function": {
                "name": t.function.name,
                "description": t.function.description,
                "parameters": t.function.parameters,
            },
        }
        for t in tools
    ]


def build_response(
    request: ChatCompletionRequest,
    content: str,
    tool_calls: list,
    finish_reason: str,
    prompt_tokens: int,
    completion_tokens: int,
) -> dict:
    msg = {"role": "assistant"}
    if tool_calls:
        msg["content"] = None
        msg["tool_calls"] = tool_calls
        finish_reason = "tool_calls"
    else:
        msg["content"] = content
    return {
        "id": f"chatcmpl-{uuid.uuid4().hex[:8]}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": request.model,
        "choices": [{"index": 0, "message": msg, "finish_reason": finish_reason}],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
    }


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


async def _handle_chat(request: ChatCompletionRequest):
    if model is None or tokenizer is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    template_messages = build_messages_for_template(request.messages)

    if request.tools and (not template_messages or template_messages[0]["role"] not in ("system", "developer")):
        template_messages.insert(0, {
            "role": "system",
            "content": "You are a helpful assistant. When you need information or need to perform an action, use the available tools. Always call a tool if one is relevant.",
        })
    tools_decl = build_tool_declarations(request.tools) if request.tools else None

    try:
        prompt = tokenizer.apply_chat_template(
            template_messages,
            tools=tools_decl,
            tokenize=False,
            add_generation_prompt=True,
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
        tool_calls = parse_tool_calls(response_text)

        if tool_calls:
            visible_text = TOOL_CALL_PATTERN.sub("", response_text).strip()
            visible_text, _ = extract_visible_content(visible_text)
            prompt_tokens = len(tokenizer.encode(prompt))
            completion_tokens = len(tokenizer.encode(response_text))
            return build_response(
                request, visible_text, tool_calls, "tool_calls",
                prompt_tokens, completion_tokens,
            )

        visible, reasoning = extract_visible_content(response_text)
        final_response = visible or reasoning or response_text
        prompt_tokens = len(tokenizer.encode(prompt))
        completion_tokens = len(tokenizer.encode(final_response))
        return build_response(
            request, final_response, [], "stop",
            prompt_tokens, completion_tokens,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    return await _handle_chat(request)


@app.post("/v1/responses")
async def responses(request: ChatCompletionRequest):
    return await _handle_chat(request)


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
