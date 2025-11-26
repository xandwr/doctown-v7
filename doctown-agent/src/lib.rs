// lib.rs - Doctown Agent library

pub mod api;
pub mod mcp;

pub use api::*;
pub use mcp::{DEFAULT_MCP_PORT, McpServer, WebSocketConfig, serve_websocket};
