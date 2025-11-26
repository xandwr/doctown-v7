#!/bin/bash

echo "Testing the 'ask' tool via MCP server..."
echo

# Test 1: List all tools to verify 'ask' is registered
echo "1. Checking if 'ask' tool is registered..."
curl -s http://localhost:8766/ \
  -X POST \
  -H "Content-Type: application/json" \
  -d '{"jsonrpc":"2.0","id":1,"method":"tools/list"}' \
  | jq -r '.result.tools[] | select(.name == "ask")'

echo
echo "---"
echo

# Test 2: Call the ask tool with a question
echo "2. Asking: 'How do I add a new MCP tool?'"
curl -s http://localhost:8766/ \
  -X POST \
  -H "Content-Type: application/json" \
  -d '{
    "jsonrpc":"2.0",
    "id":2,
    "method":"tools/call",
    "params":{
      "name":"ask",
      "arguments":{
        "question":"How do I add a new MCP tool?",
        "limit":5
      }
    }
  }' | jq '.'

echo
echo "---"
echo

# Test 3: Ask a different question
echo "3. Asking: 'What does the embedding engine do?'"
curl -s http://localhost:8766/ \
  -X POST \
  -H "Content-Type: application/json" \
  -d '{
    "jsonrpc":"2.0",
    "id":3,
    "method":"tools/call",
    "params":{
      "name":"ask",
      "arguments":{
        "question":"What does the embedding engine do?",
        "limit":3
      }
    }
  }' | jq '.result.content[0].text' -r | jq '.'
