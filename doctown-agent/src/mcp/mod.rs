// mcp/mod.rs - MCP server module

pub mod server;
pub mod sse;
pub mod websocket;

pub use server::McpServer;
pub use sse::{DEFAULT_MCP_SSE_PORT, SseConfig, serve_sse};
pub use websocket::{DEFAULT_MCP_PORT, WebSocketConfig, serve_websocket};
