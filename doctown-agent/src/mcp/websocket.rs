// mcp/websocket.rs - WebSocket server wrapper for MCP protocol
//
// Exposes the MCP server over WebSocket (ws:// or wss://) on port 8765
// following the emerging MCP spec conventions (OpenAI, Anthropic, Cursor).
//
// This allows remote agents to connect to the doctown MCP server over the network,
// making it suitable for deployment on RunPod or other cloud platforms.

use anyhow::{Context, Result};
use futures_util::{SinkExt, StreamExt};
use std::net::SocketAddr;
use std::path::PathBuf;
use std::sync::Arc;
use tokio::net::{TcpListener, TcpStream};
use tokio::sync::Mutex;
use tokio_tungstenite::{accept_async, tungstenite::Message};

use super::server::{JsonRpcRequest, JsonRpcResponse, McpServer};

/// Default MCP WebSocket port following spec conventions
pub const DEFAULT_MCP_PORT: u16 = 8765;

/// WebSocket server configuration
pub struct WebSocketConfig {
    pub host: String,
    pub port: u16,
    pub docpack_path: PathBuf,
}

impl Default for WebSocketConfig {
    fn default() -> Self {
        Self {
            host: "0.0.0.0".to_string(),
            port: DEFAULT_MCP_PORT,
            docpack_path: PathBuf::new(),
        }
    }
}

/// Start the WebSocket MCP server
pub async fn serve_websocket(config: WebSocketConfig) -> Result<()> {
    let addr: SocketAddr = format!("{}:{}", config.host, config.port)
        .parse()
        .context("Invalid host:port")?;

    let listener = TcpListener::bind(&addr)
        .await
        .context(format!("Failed to bind to {}", addr))?;

    eprintln!("MCP WebSocket server listening on: ws://{}", addr);
    eprintln!("Docpack: {}", config.docpack_path.display());
    eprintln!();
    eprintln!("Example client configuration:");
    eprintln!("  {{");
    eprintln!("    \"mcpServers\": {{");
    eprintln!("      \"doctown\": {{");
    eprintln!("        \"url\": \"ws://{}\"", addr);
    eprintln!("      }}");
    eprintln!("    }}");
    eprintln!("  }}");
    eprintln!();

    // Accept connections in a loop
    loop {
        match listener.accept().await {
            Ok((stream, peer_addr)) => {
                eprintln!("New connection from: {}", peer_addr);
                let docpack_path = config.docpack_path.clone();

                // Spawn a task to handle this connection
                tokio::spawn(async move {
                    if let Err(e) = handle_connection(stream, docpack_path, peer_addr).await {
                        eprintln!("Error handling connection from {}: {}", peer_addr, e);
                    }
                });
            }
            Err(e) => {
                eprintln!("Error accepting connection: {}", e);
            }
        }
    }
}

/// Handle a single WebSocket connection
async fn handle_connection(
    stream: TcpStream,
    docpack_path: PathBuf,
    peer_addr: SocketAddr,
) -> Result<()> {
    // Upgrade to WebSocket
    let ws_stream = accept_async(stream)
        .await
        .context("WebSocket handshake failed")?;

    eprintln!("WebSocket established with {}", peer_addr);

    let (mut ws_sender, mut ws_receiver) = ws_stream.split();

    // Create an MCP server instance for this connection
    let mcp_server = Arc::new(Mutex::new(
        McpServer::new(docpack_path).context("Failed to create MCP server")?,
    ));

    // Process messages from the client
    while let Some(msg) = ws_receiver.next().await {
        let msg = match msg {
            Ok(m) => m,
            Err(e) => {
                eprintln!("WebSocket error from {}: {}", peer_addr, e);
                break;
            }
        };

        match msg {
            Message::Text(text) => {
                // Parse JSON-RPC request
                let response = match serde_json::from_str::<JsonRpcRequest>(&text) {
                    Ok(request) => {
                        // Handle the request using the MCP server
                        let mut server = mcp_server.lock().await;
                        handle_request(&mut server, request)
                    }
                    Err(e) => JsonRpcResponse {
                        jsonrpc: "2.0".to_string(),
                        id: None,
                        result: None,
                        error: Some(super::server::JsonRpcError {
                            code: -32700,
                            message: format!("Parse error: {}", e),
                            data: None,
                        }),
                    },
                };

                // Send response back
                let response_json = serde_json::to_string(&response)?;
                ws_sender
                    .send(Message::Text(response_json.into()))
                    .await
                    .context("Failed to send response")?;
            }
            Message::Close(_) => {
                eprintln!("Client {} closed connection", peer_addr);
                break;
            }
            Message::Ping(data) => {
                // Respond to ping with pong
                ws_sender
                    .send(Message::Pong(data))
                    .await
                    .context("Failed to send pong")?;
            }
            _ => {
                // Ignore other message types (Binary, Pong, Frame)
            }
        }
    }

    eprintln!("Connection closed with {}", peer_addr);
    Ok(())
}

/// Handle a JSON-RPC request by delegating to the MCP server
fn handle_request(server: &mut McpServer, request: JsonRpcRequest) -> JsonRpcResponse {
    server.handle_request(request)
}
