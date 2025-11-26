# WebSocket MCP Server Setup

This guide explains how to run the Doctown Agent MCP server in WebSocket mode for remote access.

## Quick Start

### Local WebSocket Server

```bash
# Start on default port 8765
doctown-agent my-project.docpack --websocket

# Start on custom port
doctown-agent my-project.docpack -w -p 9000
```

### Client Configuration

Add to your agent's MCP configuration (e.g., Claude Desktop, Cursor, etc.):

```json
{
  "mcpServers": {
    "doctown": {
      "url": "ws://localhost:8765"
    }
  }
}
```

## Port 8765: The MCP Standard

Following emerging conventions from OpenAI, Anthropic, and Cursor, we use **port 8765** as the default for MCP WebSocket servers because:

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
3. Run command: `doctown-agent /data/project.docpack --websocket`
4. RunPod will assign a public IP

### 4. Connect from Client

```json
{
  "mcpServers": {
    "doctown": {
      "url": "ws://<runpod-public-ip>:8765"
    }
  }
}
```

## TLS/WSS (Production)

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

    location /mcp {
        proxy_pass http://mcp_backend;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
```

Client configuration:

```json
{
  "mcpServers": {
    "doctown": {
      "url": "wss://mcp.yourdomain.com/mcp"
    }
  }
}
```

## Architecture

```
Agent Client (Claude, Cursor, etc.)
        |
        | WebSocket (JSON-RPC 2.0)
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

1. **Connection**: Client establishes WebSocket connection
2. **Initialize**: Client sends `initialize` request
3. **Discovery**: Client calls `tools/list` to discover available tools
4. **Invocation**: Client calls `tools/call` with tool name and arguments
5. **Response**: Server returns results via JSON-RPC 2.0 format

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

### WebSocket Upgrade Failed

- Verify client is using `ws://` (not `http://`)
- Check nginx proxy settings if using reverse proxy
- Ensure WebSocket upgrade headers are preserved

## Examples

See [examples/mcp-client-config.json](./examples/mcp-client-config.json) for complete client configuration examples.

## Implementation Details

The WebSocket server wraps the existing stdio-based MCP server:

- Each WebSocket connection gets its own `McpServer` instance
- JSON-RPC messages are forwarded between WebSocket and MCP server
- Supports multiple concurrent client connections
- Automatic ping/pong for connection keepalive

For implementation details, see [src/mcp/websocket.rs](./src/mcp/websocket.rs).
