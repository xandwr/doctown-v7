// lib.rs - Doctown Agent library

pub mod api;
pub mod mcp;

pub use api::*;
pub use mcp::{
    DEFAULT_MCP_PORT, DEFAULT_MCP_SSE_PORT, McpServer, SseConfig, WebSocketConfig, serve_sse,
    serve_websocket,
};
