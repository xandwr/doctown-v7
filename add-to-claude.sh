#!/bin/bash
# add-to-claude.sh - Add the Doctown MCP server to Claude Code
#
# Usage:
#   ./add-to-claude.sh [server-name] [port]
#
# Examples:
#   ./add-to-claude.sh                    # Name: doctown, Port: 8765
#   ./add-to-claude.sh my-project         # Name: my-project, Port: 8765
#   ./add-to-claude.sh my-project 9000    # Name: my-project, Port: 9000

set -e

# Configuration
SERVER_NAME="${1:-doctown}"
PORT="${2:-8765}"
URL="ws://localhost:$PORT"

echo "Adding Doctown MCP server to Claude Code..."
echo "  Name: $SERVER_NAME"
echo "  URL: $URL"
echo ""

# Check if claude command is available
if ! command -v claude &> /dev/null; then
    echo "Error: 'claude' command not found"
    echo ""
    echo "Please install Claude Code first:"
    echo "  https://docs.anthropic.com/claude/docs/claude-code"
    exit 1
fi

# Check if server is running
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PID_FILE="$SCRIPT_DIR/logs/mcp-server.pid"

if [ ! -f "$PID_FILE" ]; then
    echo "⚠️  Warning: MCP server is not running"
    echo ""
    read -p "Start the server now? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        "$SCRIPT_DIR/start-mcp-server.sh" "" "$PORT"
        echo ""
        sleep 2
    else
        echo "Please start the server first:"
        echo "  ./start-mcp-server.sh"
        exit 1
    fi
fi

# Add to Claude Code
echo "Adding MCP server to Claude Code..."

# Use the websocket transport (Claude Code supports WebSocket MCP servers)
claude mcp add "$SERVER_NAME" "$URL"

if [ $? -eq 0 ]; then
    echo ""
    echo "✓ Successfully added '$SERVER_NAME' to Claude Code!"
    echo ""
    echo "The MCP server is now available with these tools:"
    echo "  • search_symbols - Find symbols by name or content"
    echo "  • get_impact - Analyze dependencies and impact"
    echo "  • get_dependencies - Reverse dependency analysis"
    echo "  • read_file - Read source files"
    echo "  • get_symbol_content - Get symbol source code"
    echo "  • propose_refactor - AI-assisted refactoring"
    echo "  • generate_symbol_docs - Auto-generate documentation"
    echo "  • create_test_for_symbol - Generate test cases"
    echo "  ... and more!"
    echo ""
    echo "To use, just chat with Claude Code and it will automatically"
    echo "discover and use these tools when analyzing your codebase."
    echo ""
    echo "To remove later:"
    echo "  claude mcp remove $SERVER_NAME"
else
    echo ""
    echo "✗ Failed to add MCP server to Claude Code"
    echo ""
    echo "You can manually add it to your config:"
    echo ""
    echo "  ~/.config/claude-code/config.json"
    echo ""
    echo "Add this entry:"
    echo "  {"
    echo "    \"mcpServers\": {"
    echo "      \"$SERVER_NAME\": {"
    echo "        \"url\": \"$URL\""
    echo "      }"
    echo "    }"
    echo "  }"
    exit 1
fi
