"""
MLX Server - OpenAI-compatible API server for Apple Silicon.
Supports Gemma 4 and other mlx-vlm models with TurboQuant KV cache.
"""

import gc
import os
import re
import json
import time
import uuid
from contextlib import asynccontextmanager

import mlx.core as mx
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import Optional, List, Dict, Any, Union

from mlx_vlm import load, generate, stream_generate
from mlx_vlm.models.base import load_chat_template

# Global state
model = None
processor = None
tokenizer = None
config = None

TOOL_CALL_PATTERN = re.compile(
    r"<\|tool_call\>call:(\w+)\{(.*?)\}<tool_call\|>",
    re.DOTALL,
)
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


@asynccontextmanager
async def lifespan(app: FastAPI):
    global model, processor, tokenizer, config
    print(f"Loading model: {MODEL_NAME}")
    model, processor = load(MODEL_NAME)
    tokenizer = getattr(processor, "tokenizer", processor)
    config = model.config

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
    if KV_BITS:
        print(f"TurboQuant KV cache: {KV_BITS}-bit ({KV_QUANT_SCHEME})")
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
KV_BITS = float(os.environ.get("KV_BITS", "0")) or None
KV_QUANT_SCHEME = os.environ.get("KV_QUANT_SCHEME", "turboquant")
KV_GROUP_SIZE = int(os.environ.get("KV_GROUP_SIZE", "64"))
MAX_KV_SIZE = int(os.environ.get("MAX_KV_SIZE", "0")) or None
QUANTIZED_KV_START = int(os.environ.get("QUANTIZED_KV_START", "5000"))


class FunctionDef(BaseModel):
    name: str
    description: str
    parameters: Dict[str, Any]


class ToolDef(BaseModel):
    type: str = "function"
    function: FunctionDef


class Message(BaseModel):
    role: str
    content: Optional[Union[str, List[Dict[str, Any]]]] = None
    tool_calls: Optional[List[Dict[str, Any]]] = None
    tool_call_id: Optional[str] = None


class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[Message]
    tools: Optional[List[ToolDef]] = None
    max_tokens: Optional[int] = 512
    temperature: Optional[float] = 0.7
    top_p: Optional[float] = None
    stream: Optional[bool] = False
    enable_thinking: Optional[bool] = None
    thinking_budget: Optional[int] = None


class ResponsesRequest(BaseModel):
    model: str
    input: Union[str, List[Message]]
    tools: Optional[List[ToolDef]] = None
    max_output_tokens: Optional[int] = 512
    temperature: Optional[float] = 0.7
    top_p: Optional[float] = None
    stream: Optional[bool] = False
    enable_thinking: Optional[bool] = None
    thinking_budget: Optional[int] = None


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


def build_generation_kwargs(request) -> dict:
    kwargs = {}
    if KV_BITS is not None:
        kwargs["kv_bits"] = KV_BITS
        kwargs["kv_quant_scheme"] = KV_QUANT_SCHEME
        kwargs["kv_group_size"] = KV_GROUP_SIZE
        kwargs["quantized_kv_start"] = QUANTIZED_KV_START
    if MAX_KV_SIZE is not None:
        kwargs["max_kv_size"] = MAX_KV_SIZE
    if getattr(request, "temperature", None) is not None:
        kwargs["temperature"] = request.temperature
    if getattr(request, "top_p", None) is not None:
        kwargs["top_p"] = request.top_p
    return kwargs


def build_template_kwargs(request) -> dict:
    kwargs = {}
    if getattr(request, "enable_thinking", None) is not None:
        kwargs["enable_thinking"] = request.enable_thinking
    if getattr(request, "thinking_budget", None) is not None:
        kwargs["thinking_budget"] = request.thinking_budget
    return kwargs


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


def prepare_prompt(request_messages, tools=None, extra_template_kwargs=None):
    template_messages = build_messages_for_template(request_messages)

    if tools and (not template_messages or template_messages[0].get("role") not in ("system", "developer")):
        template_messages.insert(0, {
            "role": "system",
            "content": "You are a helpful assistant. When you need information or need to perform an action, use the available tools. Always call a tool if one is relevant.",
        })

    tools_decl = build_tool_declarations(tools) if tools else None
    template_kwargs = extra_template_kwargs or {}

    try:
        prompt = tokenizer.apply_chat_template(
            template_messages,
            tools=tools_decl,
            tokenize=False,
            add_generation_prompt=True,
            **template_kwargs,
        )
    except ValueError:
        raise HTTPException(
            status_code=500,
            detail="Chat template not available. Ensure chat_template.json exists in the model directory.",
        )

    return prompt


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
    if model is None or tokenizer is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    template_kwargs = build_template_kwargs(request)
    prompt = prepare_prompt(request.messages, request.tools, template_kwargs)
    gen_kwargs = {**build_generation_kwargs(request), **template_kwargs}

    if request.stream:
        async def event_stream():
            created = int(time.time())
            role_chunk = {
                "id": f"chatcmpl-{uuid.uuid4().hex[:8]}",
                "object": "chat.completion.chunk",
                "created": created,
                "model": request.model,
                "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}],
            }
            yield f"data: {json.dumps(role_chunk, ensure_ascii=False)}\n\n"

            full_text = ""
            for response in stream_generate(
                model, processor, prompt=prompt,
                max_tokens=request.max_tokens or 512,
                verbose=False,
                **gen_kwargs,
            ):
                delta = response.text
                full_text += delta
                content_chunk = {
                    "id": f"chatcmpl-{uuid.uuid4().hex[:8]}",
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": request.model,
                    "choices": [{"index": 0, "delta": {"content": delta}, "finish_reason": None}],
                }
                yield f"data: {json.dumps(content_chunk, ensure_ascii=False)}\n\n"

            tool_calls = parse_tool_calls(full_text)
            finish_reason = "tool_calls" if tool_calls else "stop"

            finish_chunk = {
                "id": f"chatcmpl-{uuid.uuid4().hex[:8]}",
                "object": "chat.completion.chunk",
                "created": created,
                "model": request.model,
                "choices": [{"index": 0, "delta": {}, "finish_reason": finish_reason}],
                "usage": {
                    "prompt_tokens": response.prompt_tokens,
                    "completion_tokens": response.generation_tokens,
                    "total_tokens": response.prompt_tokens + response.generation_tokens,
                },
            }
            yield f"data: {json.dumps(finish_chunk, ensure_ascii=False)}\n\n"
            yield "data: [DONE]\n\n"
            mx.clear_cache()
            gc.collect()

        return StreamingResponse(event_stream(), media_type="text/event-stream")

    # Non-streaming
    try:
        result = generate(
            model, processor, prompt=prompt,
            max_tokens=request.max_tokens or 512,
            verbose=False,
            **gen_kwargs,
        )

        response_text = result.text
        tool_calls = parse_tool_calls(response_text)

        if tool_calls:
            visible_text = TOOL_CALL_PATTERN.sub("", response_text).strip()
            visible_text, _ = extract_visible_content(visible_text)
            return build_response(
                request, visible_text, tool_calls, "tool_calls",
                result.prompt_tokens, result.generation_tokens,
            )

        visible, reasoning = extract_visible_content(response_text)
        final_response = visible or reasoning or response_text
        return build_response(
            request, final_response, [], "stop",
            result.prompt_tokens, result.generation_tokens,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/v1/responses")
async def responses_endpoint(request: ResponsesRequest):
    if model is None or tokenizer is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    if isinstance(request.input, str):
        messages = [{"role": "user", "content": request.input}]
    else:
        messages = [{"role": msg.role, "content": msg.content} for msg in request.input if msg.content is not None]

    template_kwargs = build_template_kwargs(request)
    prompt = prepare_prompt(messages, request.tools, template_kwargs)
    gen_kwargs = {**build_generation_kwargs(request), **template_kwargs}

    if request.stream:
        async def event_stream():
            created = int(time.time())
            full_text = ""
            last_response = None
            for response in stream_generate(
                model, processor, prompt=prompt,
                max_tokens=request.max_output_tokens or 512,
                verbose=False,
                **gen_kwargs,
            ):
                full_text += response.text
                last_response = response

            tool_calls = parse_tool_calls(full_text)
            output_parts = []
            if tool_calls:
                for tc in tool_calls:
                    output_parts.append({
                        "type": "function_call",
                        "id": tc["id"],
                        "name": tc["function"]["name"],
                        "arguments": tc["function"]["arguments"],
                    })
                visible_text = TOOL_CALL_PATTERN.sub("", full_text).strip()
                if visible_text:
                    output_parts.append({"type": "message", "role": "assistant", "content": visible_text})
            else:
                visible, reasoning = extract_visible_content(full_text)
                final_response = visible or reasoning or full_text
                output_parts.append({"type": "message", "role": "assistant", "content": final_response})

            prompt_tokens = last_response.prompt_tokens if last_response else 0
            completion_tokens = last_response.generation_tokens if last_response else 0

            return {
                "id": f"resp_{uuid.uuid4().hex[:8]}",
                "object": "response",
                "model": request.model,
                "output": output_parts,
                "usage": {
                    "input_tokens": prompt_tokens,
                    "output_tokens": completion_tokens,
                    "total_tokens": prompt_tokens + completion_tokens,
                },
            }

            mx.clear_cache()
            gc.collect()

        return StreamingResponse(event_stream(), media_type="text/event-stream")

    # Non-streaming
    try:
        result = generate(
            model, processor, prompt=prompt,
            max_tokens=request.max_output_tokens or 512,
            verbose=False,
            **gen_kwargs,
        )

        response_text = result.text
        tool_calls = parse_tool_calls(response_text)

        output_parts = []
        if tool_calls:
            for tc in tool_calls:
                output_parts.append({
                    "type": "function_call",
                    "id": tc["id"],
                    "name": tc["function"]["name"],
                    "arguments": tc["function"]["arguments"],
                })
            visible_text = TOOL_CALL_PATTERN.sub("", response_text).strip()
            if visible_text:
                output_parts.append({"type": "message", "role": "assistant", "content": visible_text})
        else:
            visible, reasoning = extract_visible_content(response_text)
            final_response = visible or reasoning or response_text
            output_parts.append({"type": "message", "role": "assistant", "content": final_response})

        return {
            "id": f"resp_{uuid.uuid4().hex[:8]}",
            "object": "response",
            "model": request.model,
            "output": output_parts,
            "usage": {
                "input_tokens": result.prompt_tokens,
                "output_tokens": result.generation_tokens,
                "total_tokens": result.prompt_tokens + result.generation_tokens,
            },
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/v1/completions")
async def completions(request: ChatCompletionRequest):
    if model is None or tokenizer is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    prompt = request.messages[0].content if request.messages else ""
    try:
        result = generate(
            model, processor, prompt=prompt,
            max_tokens=request.max_tokens or 512,
            temperature=request.temperature or 0.7,
            verbose=False,
        )
        return {
            "id": f"cmpl-{uuid.uuid4().hex[:8]}",
            "object": "text_completion",
            "created": int(time.time()),
            "model": request.model,
            "choices": [{"text": result.text, "index": 0, "finish_reason": "stop"}],
            "usage": {
                "prompt_tokens": result.prompt_tokens,
                "completion_tokens": result.generation_tokens,
                "total_tokens": result.prompt_tokens + result.generation_tokens,
            },
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/v1/responses/chat/completions")
async def responses_chat_completions(request: ChatCompletionRequest):
    return await chat_completions(request)


@app.post("/unload")
async def unload():
    global model, processor, tokenizer, config
    if model is None:
        return {"status": "no model loaded"}
    model = None
    processor = None
    tokenizer = None
    config = None
    mx.clear_cache()
    gc.collect()
    return {"status": "model unloaded"}


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
    parser.add_argument("--kv-bits", type=float, default=None, help="KV cache bits (e.g. 3.5 for TurboQuant)")
    parser.add_argument("--kv-quant-scheme", type=str, default="turboquant", choices=["uniform", "turboquant"])
    parser.add_argument("--kv-group-size", type=int, default=64)
    parser.add_argument("--max-kv-size", type=int, default=None)
    args = parser.parse_args()

    MODEL_NAME = args.model
    if args.kv_bits is not None:
        KV_BITS = args.kv_bits
        KV_QUANT_SCHEME = args.kv_quant_scheme
        KV_GROUP_SIZE = args.kv_group_size
        if args.max_kv_size is not None:
            MAX_KV_SIZE = args.max_kv_size

    print(f"Starting MLX server on {args.host}:{args.port}")
    print(f"Model: {MODEL_NAME}")
    if KV_BITS:
        print(f"TurboQuant: {KV_BITS}-bit KV cache ({KV_QUANT_SCHEME})")
    uvicorn.run(app, host=args.host, port=args.port)
