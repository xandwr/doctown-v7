#!/usr/bin/env python3
"""Test HTTP MCP server connection"""

import requests
import json

def test_mcp_server():
    url = "http://localhost:8765/"
    
    # Test 1: Health check
    print("Test 1: Health Check")
    response = requests.get(url)
    print(f"  Status: {response.status_code}")
    print(f"  Response: {response.text}")
    assert response.status_code == 200
    print("  ✓ Passed\n")
    
    # Test 2: Initialize
    print("Test 2: Initialize Request")
    init_request = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "initialize",
        "params": {
            "protocolVersion": "2024-11-05",
            "capabilities": {},
            "clientInfo": {
                "name": "test-client",
                "version": "1.0.0"
            }
        }
    }
    
    response = requests.post(url, json=init_request)
    print(f"  Status: {response.status_code}")
    result = response.json()
    print(f"  Response: {json.dumps(result, indent=2)}")
    assert response.status_code == 200
    assert "result" in result
    assert "serverInfo" in result["result"]
    print("  ✓ Passed\n")
    
    # Test 3: List Tools
    print("Test 3: List Tools")
    tools_request = {
        "jsonrpc": "2.0",
        "id": 2,
        "method": "tools/list",
        "params": {}
    }
    
    response = requests.post(url, json=tools_request)
    print(f"  Status: {response.status_code}")
    result = response.json()
    print(f"  Tools available: {len(result.get('result', {}).get('tools', []))}")
    for tool in result.get('result', {}).get('tools', [])[:3]:
        print(f"    - {tool['name']}")
    assert response.status_code == 200
    print("  ✓ Passed\n")
    
    print("🎉 All tests passed!")

if __name__ == "__main__":
    try:
        test_mcp_server()
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
