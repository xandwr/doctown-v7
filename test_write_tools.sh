#!/bin/bash

# Test script for new write operation tools in MCP server

DOCPACK="localdoc.docpack"

echo "Testing MCP Write Operation Tools"
echo "===================================="
echo ""

# Test 1: List all tools
echo "Test 1: Count available tools"
TOOL_COUNT=$(echo '{"jsonrpc":"2.0","id":1,"method":"tools/list","params":{}}' | \
    ./target/release/doctown-agent "$DOCPACK" 2>/dev/null | \
    grep -o '"name":"[^"]*"' | wc -l)
echo "Total tools available: $TOOL_COUNT"
echo "Expected: 18 (12 read + 6 write)"
echo ""

# Test 2: Check if write tools are registered
echo "Test 2: Verify write tools are registered"
WRITE_TOOLS=$(echo '{"jsonrpc":"2.0","id":1,"method":"tools/list","params":{}}' | \
    ./target/release/doctown-agent "$DOCPACK" 2>/dev/null | \
    grep -o '"name":"[^"]*"' | grep -E "(apply_patch|propose_refactor|generate_symbol_docs|rewrite_chunk|update_file_section|create_test_for_symbol)")

echo "Write tools found:"
echo "$WRITE_TOOLS" | sed 's/"name"://g' | sed 's/"//g'
echo ""

# Test 3: Try generate_symbol_docs tool
echo "Test 3: Call generate_symbol_docs for 'main' symbol"
echo '{"jsonrpc":"2.0","id":3,"method":"tools/call","params":{"name":"generate_symbol_docs","arguments":{"symbol":"main","style":"rustdoc","include_examples":true}}}' | \
    ./target/release/doctown-agent "$DOCPACK" 2>/dev/null | \
    jq -r '.result.content[0].text' | jq -r '.symbol, .file_path, .documentation' | head -20
echo ""

# Test 4: Try propose_refactor tool
echo "Test 4: Call propose_refactor for 'main' symbol (rename)"
echo '{"jsonrpc":"2.0","id":4,"method":"tools/call","params":{"name":"propose_refactor","arguments":{"symbol":"main","refactor_type":"rename_symbol","options":{"new_name":"run_main"}}}}' | \
    ./target/release/doctown-agent "$DOCPACK" 2>/dev/null | \
    jq -r '.result.content[0].text' | jq -r '.symbol, .refactor_type, .confidence, .impact_analysis' | head -20
echo ""

# Test 5: Try create_test_for_symbol tool
echo "Test 5: Call create_test_for_symbol for 'main' symbol"
echo '{"jsonrpc":"2.0","id":5,"method":"tools/call","params":{"name":"create_test_for_symbol","arguments":{"symbol":"main","test_type":"unit"}}}' | \
    ./target/release/doctown-agent "$DOCPACK" 2>/dev/null | \
    jq -r '.result.content[0].text' | jq -r '.symbol, .test_file_path, .test_cases | length' | head -20
echo ""

echo "Tests complete!"
