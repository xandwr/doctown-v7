#!/bin/bash
# test_mcp.sh - Test the doctown-agent MCP server

DOCPACK="${1:-localdoc.docpack}"

if [ ! -f "$DOCPACK" ]; then
    echo "Error: Docpack file not found: $DOCPACK"
    echo "Usage: $0 [path-to-docpack]"
    exit 1
fi

AGENT="./target/release/doctown-agent"

if [ ! -f "$AGENT" ]; then
    echo "Building doctown-agent..."
    cargo build --release -p doctown-agent
fi

echo "Testing MCP server with: $DOCPACK"
echo ""

# Test 1: Initialize
echo "Test 1: Initialize"
echo '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{}}' | $AGENT "$DOCPACK"
echo ""

# Test 2: List tools
echo "Test 2: List tools"
echo '{"jsonrpc":"2.0","id":2,"method":"tools/list"}' | $AGENT "$DOCPACK"
echo ""

# Test 3: Get quickstart
echo "Test 3: Get quickstart"
echo '{"jsonrpc":"2.0","id":3,"method":"tools/call","params":{"name":"get_quickstart","arguments":{}}}' | $AGENT "$DOCPACK"
echo ""

# Test 4: Search symbols
echo "Test 4: Search for 'build' symbols"
echo '{"jsonrpc":"2.0","id":4,"method":"tools/call","params":{"name":"search_symbols","arguments":{"query":"build","limit":5}}}' | $AGENT "$DOCPACK"
echo ""

# Test 5: List subsystems
echo "Test 5: List subsystems"
echo '{"jsonrpc":"2.0","id":5,"method":"tools/call","params":{"name":"list_subsystems","arguments":{}}}' | $AGENT "$DOCPACK"
echo ""

echo "Tests complete!"
