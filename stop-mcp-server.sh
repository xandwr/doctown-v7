#!/bin/bash
# stop-mcp-server.sh - Stop the Doctown MCP WebSocket server

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="$SCRIPT_DIR/logs"
PID_FILE="$LOG_DIR/mcp-server.pid"

if [ ! -f "$PID_FILE" ]; then
    echo "MCP server is not running (no PID file found)"
    exit 0
fi

PID=$(cat "$PID_FILE")

if ! ps -p "$PID" > /dev/null 2>&1; then
    echo "MCP server is not running (stale PID file)"
    rm "$PID_FILE"
    exit 0
fi

echo "Stopping MCP server (PID: $PID)..."
kill "$PID"

# Wait for process to stop
for i in {1..10}; do
    if ! ps -p "$PID" > /dev/null 2>&1; then
        rm "$PID_FILE"
        echo "✓ MCP server stopped successfully"
        exit 0
    fi
    sleep 1
done

# Force kill if still running
if ps -p "$PID" > /dev/null 2>&1; then
    echo "Force killing MCP server..."
    kill -9 "$PID"
    rm "$PID_FILE"
    echo "✓ MCP server force stopped"
else
    rm "$PID_FILE"
    echo "✓ MCP server stopped successfully"
fi
