// mcp/sse.rs - SSE (Server-Sent Events) server for MCP protocol
//
// Implements the MCP specification's SSE transport for remote access.
// Uses HTTP with Server-Sent Events for server-to-client messages
// and POST requests for client-to-server messages.

use anyhow::{Context, Result};
use axum::{
    Json, Router,
    extract::State,
    routing::{get, post},
};
use std::net::SocketAddr;
use std::path::PathBuf;
use std::sync::Arc;
use tokio::sync::Mutex;

use super::server::{JsonRpcRequest, JsonRpcResponse, McpServer};

/// Default MCP HTTP port
pub const DEFAULT_MCP_SSE_PORT: u16 = 8765;

/// HTTP server configuration
pub struct SseConfig {
    pub host: String,
    pub port: u16,
    pub docpack_path: PathBuf,
}

impl Default for SseConfig {
    fn default() -> Self {
        Self {
            host: "0.0.0.0".to_string(),
            port: DEFAULT_MCP_SSE_PORT,
            docpack_path: PathBuf::new(),
        }
    }
}

/// Start the SSE MCP server
pub async fn serve_sse(config: SseConfig) -> Result<()> {
    let addr: SocketAddr = format!("{}:{}", config.host, config.port)
        .parse()
        .context("Invalid host:port")?;

    // Create a single global MCP server instance
    let mcp_server = Arc::new(Mutex::new(
        McpServer::new(config.docpack_path.clone()).context("Failed to create MCP server")?,
    ));

    let app = Router::new()
        .route("/", post(rpc_handler))
        .route("/", get(health_check))
        .with_state(mcp_server);

    eprintln!("MCP HTTP server listening on: http://{}", addr);
    eprintln!("Docpack: {}", config.docpack_path.display());
    eprintln!();
    eprintln!("Example client configuration:");
    eprintln!("  {{");
    eprintln!("    \"mcpServers\": {{");
    eprintln!("      \"doctown\": {{");
    eprintln!("        \"url\": \"http://{}\"", addr);
    eprintln!("      }}");
    eprintln!("    }}");
    eprintln!("  }}");
    eprintln!();

    let listener = tokio::net::TcpListener::bind(&addr)
        .await
        .context(format!("Failed to bind to {}", addr))?;

    axum::serve(listener, app).await.context("Server error")?;

    Ok(())
}

/// Health check endpoint
async fn health_check() -> &'static str {
    "MCP HTTP Server running"
}

/// RPC endpoint - receives JSON-RPC requests via POST
async fn rpc_handler(
    State(mcp_server): State<Arc<Mutex<McpServer>>>,
    Json(request): Json<JsonRpcRequest>,
) -> Json<JsonRpcResponse> {
    eprintln!(
        "Processing request: method={}, id={:?}",
        request.method, request.id
    );

    // Handle the request
    let mut server = mcp_server.lock().await;
    let response = server.handle_request(request);

    Json(response)
}
