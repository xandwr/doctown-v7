# Explain Like I'm 5 (ELI5) Feature

## Overview

The `explain=true` parameter for the `get_symbol` MCP command provides plain-English explanations of code symbols, making it easier to understand what code does and WHY it exists.

## How It Works

When you call `get_symbol` with `explain=true`, the system:

1. **Checks for existing LLM summary**: If the symbol already has an `llm_summary` field in the docpack, it returns that
2. **Generates on-demand**: If no LLM summary exists and ollama is available, it generates one using a local LLM
3. **Falls back gracefully**: If ollama is not available, it simply returns the symbol without an LLM summary

## Usage

### MCP JSON-RPC Call

```json
{
  "jsonrpc": "2.0",
  "id": 1,
  "method": "tools/call",
  "params": {
    "name": "get_symbol",
    "arguments": {
      "name": "EmbeddingEngine",
      "explain": true
    }
  }
}
```

### Response Structure

The response includes an `llm_summary` field:

```json
{
  "kind": "struct",
  "file": "src/embedding.rs",
  "summary": "ONNX-based embedding engine for semantic search...",
  "llm_summary": "This is the thing that turns text into numbers so we can compare meanings. It's like a translator that converts words into a mathematical form that computers can understand and compare."
}
```

## Setup Requirements

### Optional: Local LLM via Ollama

For on-demand explanation generation:

```bash
# Install ollama
curl -fsSL https://ollama.ai/install.sh | sh

# Pull a recommended model (3-4GB)
ollama pull phi3:mini

# Alternative models:
# ollama pull llama3.2:3b   # Very fast
# ollama pull qwen2.5:3b    # Good at code
```

### Without Ollama

The feature works gracefully without ollama:
- Pre-generated LLM summaries in the docpack are still returned
- Setting `explain=true` without ollama simply returns symbols without LLM summaries
- A warning is shown at server startup: "⚠️ LLM engine unavailable (explain mode disabled)"

## Implementation Details

### Code Changes

1. **`McpServer` struct** (`doctown-agent/src/mcp/server.rs`):
   - Added optional `llm_engine: Option<doctown::llm::LlmEngine>` field
   - Initializes LLM engine at startup if ollama is available
   - Gracefully handles unavailability

2. **`tool_get_symbol` method** (`doctown-agent/src/mcp/server.rs`):
   - Parses `explain` parameter (defaults to `false`)
   - If `explain=true` and no existing `llm_summary`, generates one on-demand
   - Uses the `LlmEngine::explain_symbol` method for generation

3. **MCP Tool Schema** (`doctown-agent/src/mcp/server.rs`):
   - Already had `explain` parameter defined in schema
   - Type: `boolean`
   - Description: "Return plain-English \"Explain Like I'm 5\" summary (default: false)"

### LLM Engine

The `doctown::llm::LlmEngine` provides:
- **Model**: Uses `phi3:mini` by default (3.8B params, fast, good quality)
- **Generation**: `explain_symbol()` method creates 1-2 sentence explanations
- **Prompt Engineering**: Focuses on WHAT the symbol does and WHY it exists
- **Context**: Uses symbol kind, signature, technical summary, and optional code context

## Testing

Run the test script to verify functionality:

```bash
./test_explain_feature.sh
```

This tests:
1. Symbol retrieval without `explain` parameter
2. Symbol retrieval with `explain=false`
3. Symbol retrieval with `explain=true` (generates LLM summary if ollama available)

## Benefits

### For AI Agents
- **Faster comprehension**: Plain-English summaries help agents understand code purpose quickly
- **Context awareness**: "WHY" explanations provide architectural insight beyond technical details
- **Onboarding**: Makes jumping into unfamiliar codebases much faster

### For Developers
- **Documentation quality**: Ensures symbols have human-readable explanations
- **Knowledge transfer**: Helps new team members understand design decisions
- **Code review**: Makes it easier to understand what changed and why

## Example Explanations

### Before (Technical Summary)
```
"ONNX-based embedding engine for semantic search with GPU acceleration"
```

### After (LLM Summary)
```
"This is the thing that turns text into numbers so we can compare meanings. 
It uses your GPU to do this really fast for lots of text at once."
```

## Future Enhancements

1. **Batch pre-generation**: Generate LLM summaries during docpack creation
2. **Caching**: Store generated summaries back to docpack
3. **Quality scoring**: Rate explanation quality and regenerate poor ones
4. **Custom prompts**: Allow users to customize explanation style
5. **Multi-language**: Support explanations in different languages

## Performance Notes

- **Cold start**: First LLM call takes ~2-5 seconds (model loading)
- **Subsequent calls**: ~0.5-1 second per symbol
- **Memory**: LLM model requires ~3-4GB RAM
- **GPU**: Optional but recommended for speed
- **Network**: No network required (fully local)

## Error Handling

The implementation handles several error cases gracefully:

1. **Ollama not installed**: Server starts with warning, `explain` mode disabled
2. **Model not available**: LLM engine initialization fails, logs helpful error
3. **Generation fails**: Logs error but returns symbol without LLM summary
4. **Missing symbol**: Returns `null` as usual (unchanged behavior)

All errors are logged to stderr but don't prevent normal operation.
