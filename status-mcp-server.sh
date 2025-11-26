#!/bin/bash
# status-mcp-server.sh - Check the status of the Doctown MCP WebSocket server

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="$SCRIPT_DIR/logs"
PID_FILE="$LOG_DIR/mcp-server.pid"
LOG_FILE="$LOG_DIR/mcp-server.log"

echo "Doctown MCP Server Status"
echo "========================="
echo ""

if [ ! -f "$PID_FILE" ]; then
    echo "Status: NOT RUNNING"
    echo ""
    echo "To start the server:"
    echo "  ./start-mcp-server.sh [path-to-docpack] [port]"
    exit 0
fi

PID=$(cat "$PID_FILE")

if ! ps -p "$PID" > /dev/null 2>&1; then
    echo "Status: NOT RUNNING (stale PID file)"
    echo ""
    echo "To start the server:"
    echo "  ./start-mcp-server.sh [path-to-docpack] [port]"
    exit 0
fi

echo "Status: RUNNING"
echo "  PID: $PID"
echo ""

# Try to extract port from log file
if [ -f "$LOG_FILE" ]; then
    PORT=$(grep -oP "listening on: ws://0\.0\.0\.0:\K\d+" "$LOG_FILE" | tail -1)
    if [ -n "$PORT" ]; then
        echo "  WebSocket URL: ws://localhost:$PORT"
        echo ""
    fi

    # Show last few log lines
    echo "Recent log entries:"
    echo "-------------------"
    tail -n 10 "$LOG_FILE"
    echo ""
fi

echo "Commands:"
echo "  View logs:  tail -f $LOG_FILE"
echo "  Stop:       ./stop-mcp-server.sh"
echo "  Restart:    ./stop-mcp-server.sh && ./start-mcp-server.sh"
