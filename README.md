# MLX Server

OpenAI-compatible API server for running MLX models on Apple Silicon.

## Features

- OpenAI-compatible `/v1/chat/completions` endpoint
- Tool calling support (function calls)
- Streaming responses
- Health check endpoint
- Built on `mlx-vlm` with TurboQuant KV cache support
- Bundled Gemma 4 chat template (HF models often lack one)

## Requirements

- Apple Silicon Mac (M1/M2/M3/M4)
- Python 3.12+

## Quick Start

```bash
# Clone and install
git clone https://github.com/TheSethRose/mlx-server-gemma.git
cd mlx-server-gemma
pip install -r requirements.txt

# Run with default model
python server.py

# Run with a specific model
python server.py --model mlx-community/gemma-4-26b-a4b-it-bf16 --port 8080

# Or via environment variable
MLX_MODEL=mlx-community/gemma-4-26b-a4b-it-bf16 python server.py
```

## API

### Health Check

```bash
curl http://localhost:8080/health
```

### List Models

```bash
curl http://localhost:8080/v1/models
```

### Chat Completions

```bash
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gemma-4",
    "messages": [{"role": "user", "content": "Hello!"}],
    "max_tokens": 512,
    "temperature": 0.7
  }'
```

### Streaming

```bash
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gemma-4",
    "messages": [{"role": "user", "content": "Hello!"}],
    "stream": true
  }'
```

### Tool Calling

```bash
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gemma-4",
    "messages": [{"role": "user", "content": "What is the weather in San Francisco?"}],
    "tools": [{
      "type": "function",
      "function": {
        "name": "get_weather",
        "description": "Get current weather for a location",
        "parameters": {
          "type": "object",
          "properties": {"location": {"type": "string"}},
          "required": ["location"]
        }
      }
    }],
    "max_tokens": 256,
    "temperature": 0.1
  }'
```

Response:

```json
{
  "choices": [
    {
      "message": {
        "role": "assistant",
        "content": null,
        "tool_calls": [
          {
            "id": "call_abc123",
            "type": "function",
            "function": {
              "name": "get_weather",
              "arguments": "{\"location\": \"San Francisco\"}"
            }
          }
        ]
      },
      "finish_reason": "tool_calls"
    }
  ]
}
```

To complete the tool call, send the tool response back:

```bash
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gemma-4",
    "messages": [
      {"role": "user", "content": "What is the weather in San Francisco?"},
      {"role": "assistant", "content": null, "tool_calls": [{"id": "call_abc123", "type": "function", "function": {"name": "get_weather", "arguments": "{\"location\": \"San Francisco\"}"}}]},
      {"role": "tool", "tool_call_id": "call_abc123", "content": "Sunny, 72°F"}
    ],
    "tools": [{"type": "function", "function": {"name": "get_weather", "description": "Get current weather", "parameters": {"type": "object", "properties": {"location": {"type": "string"}}, "required": ["location"]}}}],
    "max_tokens": 256
  }'
```

## Environment Variables

| Variable    | Default                                 | Description                  |
| ----------- | --------------------------------------- | ---------------------------- |
| `MLX_MODEL` | `mlx-community/gemma-4-26b-a4b-it-bf16` | HuggingFace model ID to load |

## Running as a Service (launchd)

```xml
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>com.local.mlx-server</string>
    <key>ProgramArguments</key>
    <array>
        <string>/path/to/venv/bin/python</string>
        <string>-m</string>
        <string>uvicorn</string>
        <string>server:app</string>
        <string>--host</string>
        <string>0.0.0.0</string>
        <string>--port</string>
        <string>8080</string>
    </array>
    <key>EnvironmentVariables</key>
    <dict>
        <key>MLX_MODEL</key>
        <string>mlx-community/gemma-4-26b-a4b-it-bf16</string>
    </dict>
    <key>RunAtLoad</key>
    <true/>
    <key>KeepAlive</key>
    <true/>
    <key>WorkingDirectory</key>
    <string>/path/to/mlx-server-gemma</string>
</dict>
</plist>
```

Save to `~/Library/LaunchAgents/com.local.mlx-server.plist`, then:

```bash
launchctl load ~/Library/LaunchAgents/com.local.mlx-server.plist
```

## Recommended Models

| Model                                                                                   | Size  | Quality      | M3 128GB Speed |
| --------------------------------------------------------------------------------------- | ----- | ------------ | -------------- |
| [gemma-4-26b-a4b-it-bf16](https://huggingface.co/mlx-community/gemma-4-26b-a4b-it-bf16) | ~26GB | Maximum      | 30-45 tok/s    |
| [gemma-4-26b-a4b-it-8bit](https://huggingface.co/mlx-community/gemma-4-26b-a4b-it-8bit) | ~8GB  | Near-perfect | 45-60 tok/s    |
| [gemma-4-31b-it-4bit](https://huggingface.co/mlx-community/gemma-4-31b-it-4bit)         | ~5GB  | Good         | 45-65 tok/s    |

Full collection: [mlx-community/gemma-4](https://huggingface.co/collections/mlx-community/gemma-4)

---

## Connect Your Apps

The server exposes an OpenAI-compatible API at `http://localhost:8080/v1`. Any app that supports a custom OpenAI base URL can use it.

### VS Code — Continue

1. Install the [Continue](https://marketplace.visualstudio.com/items?itemName=Continue.continue) extension
2. Open `~/.continue/config.yaml`
3. Add a custom OpenAI provider:

```yaml
models:
  - name: Gemma 4 (Local)
    provider: openai
    model: gemma-4
    apiBase: http://localhost:8080/v1
    roles:
      - chat
      - edit
      - apply
```

### VS Code — Cline

1. Install the [Cline](https://marketplace.visualstudio.com/items?itemName=saoudrizwan.claude-dev) extension
2. Click the Cline icon in the sidebar
3. Select **OpenAI Compatible** as the provider
4. Set:
   - **Base URL**: `http://localhost:8080/v1`
   - **Model**: `gemma-4`
   - **API Key**: `not-needed` (any value works)

### VS Code — Copilot Chat (via LM Studio proxy)

If you want to use GitHub Copilot Chat with a local model, the easiest path is to point Copilot's custom endpoint at this server. However, Copilot doesn't support arbitrary OpenAI endpoints. Use **Continue** or **Cline** instead.

### Cursor

1. Open **Cursor Settings** (`Cmd + ,`) → **AI Providers**
2. Select **OpenAI Compatible**
3. Set:
   - **API Endpoint**: `http://localhost:8080/v1`
   - **API Key**: `not-needed`
4. Set your model to `gemma-4`

### Windsurf (Codeium)

1. Open **Windsurf Settings** → **AI Provider**
2. Select **OpenAI Compatible**
3. Set:
   - **Base URL**: `http://localhost:8080/v1`
   - **Model**: `gemma-4`
   - **API Key**: `not-needed`

### Claude Desktop (via proxy)

Claude Desktop doesn't natively support custom OpenAI endpoints. Use **Continue** in VS Code or **Cursor** instead.

### LM Studio

LM Studio is an alternative to this server, not a client. If you're already running this server, you don't need LM Studio. But if you want to test models before adding them here:

1. Open LM Studio → Local Server
2. Point it at the same HuggingFace model
3. Compare performance

### Any OpenAI-compatible Client

Use these values:

| Setting      | Value                      |
| ------------ | -------------------------- |
| **Base URL** | `http://localhost:8080/v1` |
| **API Key**  | `not-needed` (any string)  |
| **Model**    | `gemma-4`                  |

## License

MIT
