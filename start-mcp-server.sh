#!/bin/bash
# start-mcp-server.sh - Start the Doctown MCP WebSocket server as a background service
#
# Usage:
#   ./start-mcp-server.sh [path-to-docpack] [port]
#
# Examples:
#   ./start-mcp-server.sh                        # Uses default docpack and port 8765
#   ./start-mcp-server.sh my-project.docpack     # Uses custom docpack, port 8765
#   ./start-mcp-server.sh my-project.docpack 9000 # Uses custom docpack and port

set -e

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DOCPACK="${1:-$SCRIPT_DIR/doctown-v7.docpack}"
PORT="${2:-8765}"
LOG_DIR="$SCRIPT_DIR/logs"
PID_FILE="$LOG_DIR/mcp-server.pid"
LOG_FILE="$LOG_DIR/mcp-server.log"

# Create logs directory
mkdir -p "$LOG_DIR"

# Check if docpack exists
if [ ! -f "$DOCPACK" ]; then
    echo "Error: Docpack not found: $DOCPACK"
    echo ""
    echo "Usage: $0 [path-to-docpack] [port]"
    echo ""
    echo "You can build a docpack with:"
    echo "  cargo run --release -- pack /path/to/your/project"
    exit 1
fi

# Check if server is already running
if [ -f "$PID_FILE" ]; then
    OLD_PID=$(cat "$PID_FILE")
    if ps -p "$OLD_PID" > /dev/null 2>&1; then
        echo "MCP server is already running (PID: $OLD_PID)"
        echo "To stop it, run: ./stop-mcp-server.sh"
        exit 1
    else
        echo "Removing stale PID file..."
        rm "$PID_FILE"
    fi
fi

# Build the server if needed
echo "Building doctown-agent..."
cargo build --release -p doctown-agent

# Start the server in the background (use SSE mode for MCP spec compliance)
echo "Starting MCP SSE server..."
echo "  Docpack: $DOCPACK"
echo "  Port: $PORT"
echo "  Log file: $LOG_FILE"

nohup "$SCRIPT_DIR/target/release/doctown-agent" "$DOCPACK" --sse --port "$PORT" \
    > "$LOG_FILE" 2>&1 &

SERVER_PID=$!
echo $SERVER_PID > "$PID_FILE"

# Wait a moment and check if it started successfully
sleep 2

if ps -p "$SERVER_PID" > /dev/null 2>&1; then
    echo ""
    echo "✓ MCP server started successfully!"
    echo "  PID: $SERVER_PID"
    echo "  HTTP URL: http://localhost:$PORT"
    echo ""
    echo "To view logs:"
    echo "  tail -f $LOG_FILE"
    echo ""
    echo "To stop the server:"
    echo "  ./stop-mcp-server.sh"
    echo ""
    echo "Add this to your agent config:"
    echo "  {"
    echo "    \"mcpServers\": {"
    echo "      \"doctown\": {"
    echo "        \"url\": \"http://localhost:$PORT\""
    echo "      }"
    echo "    }"
    echo "  }"
else
    echo ""
    echo "✗ Failed to start MCP server"
    echo "Check the log file for errors: $LOG_FILE"
    rm "$PID_FILE"
    exit 1
fi
