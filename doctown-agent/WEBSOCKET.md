# Remote MCP Server Setup

This guide explains how to run the Doctown Agent MCP server for remote access over HTTP.

## Quick Start

### Local HTTP Server

```bash
# Start on default port 8765 (recommended)
doctown-agent my-project.docpack --sse

# Start on custom port
doctown-agent my-project.docpack --sse -p 9000
```

### Client Configuration

Add to your MCP client configuration:

**For Claude CLI:**
```bash
claude mcp add --transport http doctown http://localhost:8765
```

**For Claude Desktop/Cursor:**
```json
{
  "mcpServers": {
    "doctown": {
      "url": "http://localhost:8765"
    }
  }
}
```

## Port 8765: The MCP Standard

Following emerging conventions from OpenAI, Anthropic, and Cursor, we use **port 8765** as the default for MCP servers because:

- **Not privileged**: Doesn't require root/admin permissions
- **Rarely used**: Minimal conflict with other services
- **Easy to remember**: 8765 is sequential and memorable
- **Safe on cloud platforms**: Works on RunPod, AWS, GCP, etc.

## Deployment on RunPod

### 1. Create Dockerfile

```dockerfile
FROM rust:1.75 AS builder

WORKDIR /app
COPY . .
RUN cargo build --release

FROM debian:bookworm-slim
RUN apt-get update && apt-get install -y ca-certificates && rm -rf /var/lib/apt/lists/*

COPY --from=builder /app/target/release/doctown-agent /usr/local/bin/
COPY --from=builder /app/target/release/doctown /usr/local/bin/

EXPOSE 8765

ENTRYPOINT ["doctown-agent"]
```

### 2. Build and Push to Registry

```bash
docker build -t your-registry/doctown-agent:latest .
docker push your-registry/doctown-agent:latest
```

### 3. Deploy on RunPod

1. Create a new pod with `--network=public`
2. Mount your docpack file or build it on startup
3. Run command: `doctown-agent /data/project.docpack --sse`
4. RunPod will assign a public IP

### 4. Connect from Client

```json
{
  "mcpServers": {
    "doctown": {
      "url": "http://<runpod-public-ip>:8765"
    }
  }
}
```

## TLS/HTTPS (Production)

For production deployments, add a reverse proxy (nginx, caddy) with TLS:

### Nginx Configuration

```nginx
upstream mcp_backend {
    server 127.0.0.1:8765;
}

server {
    listen 443 ssl http2;
    server_name mcp.yourdomain.com;

    ssl_certificate /etc/letsencrypt/live/mcp.yourdomain.com/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/mcp.yourdomain.com/privkey.pem;

    location / {
        proxy_pass http://mcp_backend;
        proxy_http_version 1.1;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}
```

Client configuration:

```json
{
  "mcpServers": {
    "doctown": {
      "url": "https://mcp.yourdomain.com"
    }
  }
}
```

## Architecture

```
Agent Client (Claude, Cursor, etc.)
        |
        | HTTP POST (JSON-RPC 2.0)
        |
        v
Doctown Agent (Port 8765)
        |
        | Local file access
        |
        v
  .docpack file
```

### Message Flow

1. **Initialize**: Client sends HTTP POST with `initialize` request
2. **Discovery**: Client calls `tools/list` to discover available tools
3. **Invocation**: Client calls `tools/call` with tool name and arguments
4. **Response**: Server returns results via JSON-RPC 2.0 in HTTP response

## Security Considerations

### For Development

- Bind to `127.0.0.1` for local-only access:
  ```bash
  # Modify WebSocketConfig in code to use:
  host: "127.0.0.1"
  ```

### For Production

1. **Authentication**: Add token-based auth to WebSocket handshake
2. **TLS**: Always use WSS (not WS) in production
3. **Firewall**: Restrict access to known IP ranges
4. **Rate Limiting**: Add rate limits to prevent abuse
5. **Monitoring**: Log all connections and requests

## Troubleshooting

### Connection Refused

```bash
# Check if server is running
netstat -an | grep 8765

# Check firewall rules
sudo ufw status
```

### Can't Connect from Remote

```bash
# Ensure binding to 0.0.0.0 (not 127.0.0.1)
# Check cloud provider security groups/firewall
# Verify port 8765 is exposed in container
```

### Health Check Fails

- Verify server is running: `curl http://localhost:8765/`
- Should respond with "MCP HTTP Server running"
- Check logs for errors: `tail -f logs/mcp-server.log`

## Examples

See [examples/mcp-client-config.json](./examples/mcp-client-config.json) for complete client configuration examples.

## Implementation Details

The HTTP server implements the MCP specification's HTTP transport:

- Single global `McpServer` instance handles all requests
- JSON-RPC requests sent via HTTP POST to `/`
- Stateless request/response model
- Health check endpoint at GET `/`

For implementation details, see [src/mcp/sse.rs](./src/mcp/sse.rs).

### WebSocket Mode (Legacy)

A WebSocket mode is also available for compatibility:
```bash
doctown-agent my-project.docpack --websocket
```

Note: WebSocket is not part of the official MCP specification and may not work with all MCP clients.
