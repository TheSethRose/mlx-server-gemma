#!/usr/bin/env python3
"""
Test script for MLX Server tool calling.
Run: python test_tool_calls.py
"""

import json
import re
import os
from mlx_vlm import load, generate
from mlx_vlm.models.base import load_chat_template
import pathlib

MODEL = os.environ.get("MLX_MODEL", "mlx-community/gemma-4-26b-a4b-it-bf16")

TOOL_CALL_RE = re.compile(r"<\|tool_call\>call:(\w+)\{(.*?)\}<tool_call\|>", re.DOTALL)


def parse_tool_calls(text):
    matches = list(TOOL_CALL_RE.finditer(text))
    if not matches:
        return []
    tool_calls = []
    for m in matches:
        name = m.group(1)
        args_str = m.group(2).strip()
        cleaned = args_str.replace('<|"|>', '"')
        cleaned = re.sub(r"(\w+):", r'"\1":', cleaned)
        try:
            args = json.loads("{" + cleaned + "}")
        except json.JSONDecodeError:
            args = {"_raw": args_str}
        tool_calls.append({"name": name, "arguments": args})
    return tool_calls


def strip_tool_calls(text):
    return TOOL_CALL_RE.sub("", text).strip()


def main():
    print(f"Loading {MODEL}...")
    model, processor = load(MODEL)
    tokenizer = getattr(processor, "tokenizer", processor)

    if not getattr(tokenizer, "chat_template", None):
        template_path = os.path.join(os.path.dirname(__file__), "chat_template.json")
        if os.path.exists(template_path):
            load_chat_template(tokenizer, pathlib.Path(template_path).parent)
            print("Chat template loaded.")
        else:
            print("ERROR: No chat template found.")
            return

    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get current weather for a location",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {"type": "string", "description": "City name"}
                    },
                    "required": ["location"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "read_file",
                "description": "Read contents of a file",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string", "description": "File path to read"}
                    },
                    "required": ["path"],
                },
            },
        },
    ]

    test_cases = [
        {
            "name": "Weather query",
            "messages": [
                {"role": "system", "content": "You are a helpful assistant. When you need information or need to perform an action, use the available tools. Always call a tool if one is relevant."},
                {"role": "user", "content": "What is the weather in San Francisco?"},
            ],
        },
        {
            "name": "File read request",
            "messages": [
                {"role": "system", "content": "You are a helpful assistant. When you need information or need to perform an action, use the available tools. Always call a tool if one is relevant."},
                {"role": "user", "content": "Can you read the file /etc/hosts for me?"},
            ],
        },
        {
            "name": "No tool needed",
            "messages": [
                {"role": "system", "content": "You are a helpful assistant. When you need information or need to perform an action, use the available tools. Always call a tool if one is relevant."},
                {"role": "user", "content": "What is 2+2?"},
            ],
        },
    ]

    for tc in test_cases:
        print(f"\n{'='*60}")
        print(f"TEST: {tc['name']}")
        print(f"{'='*60}")

        prompt = tokenizer.apply_chat_template(
            tc["messages"], tools=tools, tokenize=False, add_generation_prompt=True
        )

        print(f"\nPrompt (last 300 chars):")
        print(f"...{prompt[-300:]}")

        result = generate(
            model, processor, prompt=prompt, max_tokens=200, temperature=0.1, verbose=True
        )

        text = result.text
        print(f"\nRaw output:\n{text}")

        tool_calls = parse_tool_calls(text)
        if tool_calls:
            print(f"\n*** TOOL CALLS DETECTED ({len(tool_calls)}) ***")
            for tc_item in tool_calls:
                print(f"  Name: {tc_item['name']}")
                print(f"  Args: {json.dumps(tc_item['arguments'], indent=2)}")
            remaining = strip_tool_calls(text)
            if remaining:
                print(f"\nRemaining text: {remaining}")
        else:
            print("\nNo tool calls detected.")


if __name__ == "__main__":
    main()
