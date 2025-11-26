#!/usr/bin/env python3
"""Test the new search_by_signature MCP tool"""

import json
import requests

MCP_URL = "http://localhost:8765"

def call_mcp_tool(tool_name, arguments):
    """Call an MCP tool via HTTP"""
    request_data = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "tools/call",
        "params": {
            "name": tool_name,
            "arguments": arguments
        }
    }

    response = requests.post(MCP_URL, json=request_data)
    return response.json()

# Test 1: Search for functions that return Result
print("=" * 80)
print("Test 1: Search for functions returning Result")
print("=" * 80)
result = call_mcp_tool("search_by_signature", {
    "query": "fn new() -> Result",
    "limit": 5
})
print(json.dumps(result, indent=2))

print("\n" + "=" * 80)
print("Test 2: Search for Iterator implementations")
print("=" * 80)
result = call_mcp_tool("search_by_signature", {
    "query": "impl Iterator for",
    "limit": 5
})
print(json.dumps(result, indent=2))

print("\n" + "=" * 80)
print("Test 3: Search for functions with String parameter")
print("=" * 80)
result = call_mcp_tool("search_by_signature", {
    "query": "fn(&self, String)",
    "limit": 5
})
print(json.dumps(result, indent=2))
