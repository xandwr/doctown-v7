# MCP Server Quick Start

## One-Step Setup

```bash
# Start the server and add it to Claude Code
./start-mcp-server.sh && ./add-to-claude.sh
```

Done! Claude Code can now access your docpack. 🎉

## Individual Scripts

Four simple scripts to manage your Doctown MCP WebSocket server:

## Start the Server

```bash
./start-mcp-server.sh
```

This will:
- Build the doctown-agent if needed
- Start the WebSocket server on port 8765
- Run it in the background with logging
- Show you the connection URL

**Custom options:**
```bash
./start-mcp-server.sh my-project.docpack      # Custom docpack
./start-mcp-server.sh my-project.docpack 9000 # Custom port
```

## Check Status

```bash
./status-mcp-server.sh
```

Shows:
- Whether the server is running
- Process ID and WebSocket URL
- Recent log entries

## Add to Claude Code

```bash
./add-to-claude.sh [server-name] [port]
```

Registers the MCP server with Claude Code. Examples:
```bash
./add-to-claude.sh                    # Name: doctown, Port: 8765
./add-to-claude.sh my-project         # Custom name
./add-to-claude.sh my-project 9000    # Custom port
```

## Stop the Server

```bash
./stop-mcp-server.sh
```

Gracefully stops the background server.

## View Live Logs

```bash
tail -f logs/mcp-server.log
```

## Manual Configuration (Alternative)

If you prefer to manually configure, add to `~/.config/claude-code/config.json`:

```json
{
  "mcpServers": {
    "doctown": {
      "url": "ws://localhost:8765"
    }
  }
}
```

Or use the Claude CLI:
```bash
claude mcp add doctown ws://localhost:8765
```

## Troubleshooting

### Server won't start?

1. Check if a docpack exists:
   ```bash
   ls -lh *.docpack
   ```

2. Build one if needed:
   ```bash
   cargo run --release -- pack /path/to/your/project
   ```

### Port already in use?

Start on a different port:
```bash
./start-mcp-server.sh my-project.docpack 9000
```

### Can't connect from agent?

1. Check the server is running:
   ```bash
   ./status-mcp-server.sh
   ```

2. Check logs for errors:
   ```bash
   tail -f logs/mcp-server.log
   ```

3. Verify the port matches your config:
   ```bash
   netstat -an | grep 8765
   ```

## Next Steps

Once connected, your agent can:
- Search symbols: `search_symbols`
- Analyze dependencies: `get_impact`, `get_dependencies`
- Read files: `read_file`, `get_symbol_content`
- Navigate code: `get_quickstart`, `list_subsystems`
- Modify code: `propose_refactor`, `generate_symbol_docs`

See the full API in [doctown-agent/README.md](doctown-agent/README.md).
