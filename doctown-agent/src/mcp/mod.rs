// mcp/mod.rs - MCP server module

pub mod server;
pub mod websocket;

pub use server::McpServer;
pub use websocket::{DEFAULT_MCP_PORT, WebSocketConfig, serve_websocket};
