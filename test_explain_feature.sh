#!/bin/bash
# Test script for the explain=true feature in get_symbol MCP command

set -e

DOCPACK="./doctown-v7.docpack"
MCP_SERVER="./target/release/doctown-agent"

echo "🧪 Testing explain=true feature for get_symbol"
echo "================================================"
echo ""

# Build the project if needed
if [ ! -f "$MCP_SERVER" ]; then
    echo "Building doctown-agent..."
    cargo build --release --package doctown-agent
    echo ""
fi

# Test 1: Get symbol WITHOUT explain
echo "📝 Test 1: get_symbol WITHOUT explain parameter"
echo '{"jsonrpc":"2.0","id":1,"method":"tools/call","params":{"name":"get_symbol","arguments":{"name":"McpServer"}}}' | \
    "$MCP_SERVER" "$DOCPACK" 2>&1 | jq -r '.result.content[0].text' | jq '.llm_summary' || true
echo ""

# Test 2: Get symbol WITH explain=false
echo "📝 Test 2: get_symbol WITH explain=false"
echo '{"jsonrpc":"2.0","id":2,"method":"tools/call","params":{"name":"get_symbol","arguments":{"name":"McpServer","explain":false}}}' | \
    "$MCP_SERVER" "$DOCPACK" 2>&1 | jq -r '.result.content[0].text' | jq '.llm_summary' || true
echo ""

# Test 3: Get symbol WITH explain=true (should generate LLM summary if ollama is available)
echo "📝 Test 3: get_symbol WITH explain=true"
echo '{"jsonrpc":"2.0","id":3,"method":"tools/call","params":{"name":"get_symbol","arguments":{"name":"McpServer","explain":true}}}' | \
    "$MCP_SERVER" "$DOCPACK" 2>&1 | jq -r '.result.content[0].text' | jq '.llm_summary' || true
echo ""

echo "✅ Test complete!"
echo ""
echo "Note: If LLM summaries show 'null', it means either:"
echo "  1. Ollama is not installed/running"
echo "  2. The symbol already has no llm_summary in the docpack"
echo "  3. LLM generation failed"
echo ""
echo "To enable LLM explanations, install ollama:"
echo "  curl -fsSL https://ollama.ai/install.sh | sh"
echo "  ollama pull phi3:mini"
