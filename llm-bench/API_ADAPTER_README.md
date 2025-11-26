# LLM Benchmark API Adapter

This benchmark tool now supports multiple API formats through an adapter pattern, making it compatible with both Ollama and OpenAI-style APIs.

## Supported API Formats

### 1. Ollama API (Default)
- **Endpoint**: `/api/generate`
- **Format**: Ollama-specific JSON format
- **Use case**: Testing Ollama-hosted models

### 2. OpenAI Completions API
- **Endpoint**: `/v1/completions`
- **Format**: OpenAI completions format (text completion)
- **Use case**: Testing vLLM, FastChat, or other OpenAI-compatible servers

### 3. OpenAI Chat API
- **Endpoint**: `/v1/chat/completions`
- **Format**: OpenAI chat format (message-based)
- **Use case**: Testing chat models on OpenAI-compatible servers

## Usage Examples

### Test Ollama API (Default)
```bash
python llm-bench.py
# OR
python llm-bench.py --api-type ollama --host localhost --port 11434 --model gpt-oss:20b
```

### Test OpenAI Completions API
```bash
python llm-bench.py --api-type openai --host localhost --port 8000 --model gpt-oss-20b
```

### Test OpenAI Chat API
```bash
python llm-bench.py --api-type openai_chat --host localhost --port 8000 --model gpt-oss-20b
```

### Test Remote vLLM Server
```bash
python llm-bench.py --api-type openai --host api.example.com --port 443 --model meta-llama/Llama-2-7b-hf
```

## Command Line Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--api-type` | string | `ollama` | API format: `ollama`, `openai`, or `openai_chat` |
| `--host` | string | `localhost` | API server hostname |
| `--port` | int | `11434` | API server port |
| `--model` | string | `gpt-oss:20b` | Model name/identifier |

## API Request Format Examples

### Ollama Format
```json
{
  "model": "gpt-oss:20b",
  "prompt": "Write about AI...",
  "stream": false,
  "options": {
    "temperature": 0.7,
    "top_p": 0.9,
    "top_k": 40,
    "num_predict": 500
  }
}
```

### OpenAI Completions Format
```json
{
  "model": "gpt-oss-20b",
  "prompt": "Write about AI...",
  "max_tokens": 500,
  "temperature": 0.7,
  "top_p": 0.9,
  "stream": false
}
```

### OpenAI Chat Format
```json
{
  "model": "gpt-oss-20b",
  "messages": [
    {"role": "user", "content": "Write about AI..."}
  ],
  "max_tokens": 500,
  "temperature": 0.7,
  "top_p": 0.9,
  "stream": false
}
```

## Architecture

The tool uses the **Adapter Pattern** to support multiple API formats:

```
MaxTokensBenchmark
    |
    +-- APIAdapter (abstract)
            |
            +-- OllamaAdapter
            +-- OpenAIAdapter
            +-- OpenAIChatAdapter
```

Each adapter implements:
- `build_request()`: Creates API-specific request payload
- `get_endpoint()`: Returns the correct endpoint path
- `extract_response()`: Parses API-specific response format

## Adding New API Formats

To add support for a new API format:

1. Create a new adapter class inheriting from `APIAdapter`
2. Implement the three required methods
3. Add the new API type to the `APIType` enum
4. Update the adapter selection in `MaxTokensBenchmark.__init__()`

Example:
```python
class CustomAPIAdapter(APIAdapter):
    def build_request(self, model, prompt, max_tokens, temperature, top_p, top_k):
        # Build custom request format
        return {...}

    def get_endpoint(self, base_url):
        return f"{base_url}/custom/endpoint"

    def extract_response(self, response_json):
        return response_json['custom_field']
```

## Compatibility Notes

### vLLM Compatibility
vLLM provides OpenAI-compatible endpoints. Use `--api-type openai` or `--api-type openai_chat`:
```bash
# For vLLM running on port 8000
python llm-bench.py --api-type openai_chat --port 8000 --model meta-llama/Llama-2-7b-hf
```

### FastChat Compatibility
FastChat also provides OpenAI-compatible endpoints:
```bash
python llm-bench.py --api-type openai_chat --port 8000 --model vicuna-7b-v1.5
```

### Text Generation Inference (TGI)
For Hugging Face TGI servers, you may need to create a custom adapter as TGI uses a different API format.

## Output

The benchmark results are saved to a JSON file with the format:
```
max_tokens_benchmark_{api_type}_{timestamp}.json
```

This allows you to easily compare performance across different API implementations.
