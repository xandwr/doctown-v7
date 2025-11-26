# Doctown Agent MCP Server

A Model Context Protocol (MCP) server that exposes doctown's `.docpack` files as a queryable API for AI agents and tools.

## Overview

The doctown-agent provides read-only access to all the rich data inside a `.docpack` archive through a standardized MCP interface. This allows AI coding assistants, IDEs, and other tools to understand codebases deeply without requiring direct file system access.

## Architecture

```
doctown-agent/
  api/                    # Your agent capabilities
    search.rs            # Symbol search (semantic + keyword)
    graph.rs             # Dependency analysis & graph traversal
    subsystems.rs        # Community/module queries
    tasks.rs             # Task-oriented views
    editor.rs            # File & chunk reading
  mcp/
    server.rs            # JSON-RPC routing over stdio
  manifest.json          # Tool definitions for clients
```

## Features

The MCP server exposes the following capabilities:

### Search & Discovery
- **search_symbols**: Find symbols by name or content with filtering
- **get_symbol**: Get detailed info about a specific symbol
- **list_files**: List all source files in the codebase

### Graph Analysis
- **get_impact**: Forward impact analysis (who depends on this?)
- **get_dependencies**: Reverse dependency analysis (what does this depend on?)
- **find_path**: Find shortest path between two symbols

### Subsystems
- **list_subsystems**: View detected communities/modules
- **get_subsystem**: Get detailed subsystem information

### Navigation
- **get_quickstart**: Entry points and navigation hints
- **get_task_view**: Get symbols relevant to a task
- **read_file**: Read complete file contents
- **get_symbol_content**: Get source code for a symbol

## Usage

### Running the Server

```bash
# Build the agent
cargo build --release -p doctown-agent

# Run the MCP server
./target/release/doctown-agent path/to/your.docpack
```

The server communicates over stdin/stdout using JSON-RPC 2.0.

### Example MCP Client Configuration

For Claude Desktop or other MCP clients, add to your config:

```json
{
  "mcpServers": {
    "doctown": {
      "command": "/path/to/doctown-agent",
      "args": ["/path/to/your.docpack"]
    }
  }
}
```

### Example Tool Calls

#### Search for symbols
```json
{
  "jsonrpc": "2.0",
  "id": 1,
  "method": "tools/call",
  "params": {
    "name": "search_symbols",
    "arguments": {
      "query": "parse",
      "limit": 10,
      "filter_kind": "function"
    }
  }
}
```

#### Get impact analysis
```json
{
  "jsonrpc": "2.0",
  "id": 2,
  "method": "tools/call",
  "params": {
    "name": "get_impact",
    "arguments": {
      "symbol": "build_agent_index",
      "max_depth": 3
    }
  }
}
```

#### Get quickstart guide
```json
{
  "jsonrpc": "2.0",
  "id": 3,
  "method": "tools/call",
  "params": {
    "name": "get_quickstart",
    "arguments": {}
  }
}
```

## API Design Philosophy

All APIs are **read-only** and **safe**:
- No file system writes
- No code execution
- No external network calls
- Pure data queries over the `.docpack` archive

The `.docpack` contains everything needed:
- Pre-computed embeddings
- Community detection results
- Symbol tables and graph structures
- Source code snapshots
- Documentation and summaries

## Development

### Adding New Tools

1. Implement the logic in `api/` modules
2. Add the tool handler in `mcp/server.rs`
3. Update `manifest.json` with tool definition
4. Document in this README

### Testing

```bash
# Build and test
cargo test -p doctown-agent

# Test with a real docpack
cargo run -p doctown-agent -- test.docpack
```

Then send JSON-RPC requests via stdin to test the server.

## Protocol

The server implements MCP (Model Context Protocol) version 2024-11-05:

- **Transport**: JSON-RPC 2.0 over stdio
- **Methods**:
  - `initialize`: Server capabilities
  - `tools/list`: Available tools
  - `tools/call`: Invoke a tool

See `manifest.json` for complete tool schemas.

## License

Same as doctown main project.
