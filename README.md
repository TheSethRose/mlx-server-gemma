# MLX Server

OpenAI-compatible API server for running MLX models on Apple Silicon.

## Features

- OpenAI-compatible `/v1/chat/completions` endpoint
- Streaming responses
- Health check endpoint
- Built on `mlx-vlm` with TurboQuant KV cache support
- Bundled Gemma 4 chat template (models on HuggingFace often lack one)

## Requirements

- Apple Silicon Mac (M1/M2/M3/M4)
- Python 3.12+

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run with default model (Gemma 4 26B-A4B bf16)
python server.py

# Run with a specific model
python server.py --model mlx-community/gemma-4-26b-a4b-bf16 --port 8080

# Or set via environment variable
MLX_MODEL=mlx-community/gemma-4-26b-a4b-bf16 python server.py
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

## Environment Variables

| Variable    | Default                              | Description                  |
| ----------- | ------------------------------------ | ---------------------------- |
| `MLX_MODEL` | `mlx-community/gemma-4-26b-a4b-bf16` | HuggingFace model ID to load |

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
        <string>mlx-community/gemma-4-26b-a4b-bf16</string>
    </dict>
    <key>RunAtLoad</key>
    <true/>
    <key>KeepAlive</key>
    <true/>
    <key>WorkingDirectory</key>
    <string>/path/to/mlx-server</string>
</dict>
</plist>
```

Save to `~/Library/LaunchAgents/com.local.mlx-server.plist`, then:

```bash
launchctl load ~/Library/LaunchAgents/com.local.mlx-server.plist
```

## Recommended Models for Apple Silicon

| Model                                                                                   | Size  | Quality      | M3 128GB Speed |
| --------------------------------------------------------------------------------------- | ----- | ------------ | -------------- |
| [gemma-4-26b-a4b-bf16](https://huggingface.co/mlx-community/gemma-4-26b-a4b-bf16)       | ~26GB | Maximum      | 30-45 tok/s    |
| [gemma-4-26b-a4b-it-8bit](https://huggingface.co/mlx-community/gemma-4-26b-a4b-it-8bit) | ~8GB  | Near-perfect | 45-60 tok/s    |
| [gemma-4-31b-it-4bit](https://huggingface.co/mlx-community/gemma-4-31b-it-4bit)         | ~5GB  | Good         | 45-65 tok/s    |

Full collection: [mlx-community/gemma-4](https://huggingface.co/collections/mlx-community/gemma-4)

## License

MIT
