// main.rs - Doctown Agent MCP server entry point

use anyhow::Result;
use doctown_agent::{
    DEFAULT_MCP_PORT, DEFAULT_MCP_SSE_PORT, McpServer, SseConfig, WebSocketConfig, serve_sse,
    serve_websocket,
};
use std::path::PathBuf;

#[tokio::main]
async fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();

    if args.len() < 2 {
        print_usage();
        std::process::exit(1);
    }

    let docpack_path = PathBuf::from(&args[1]);

    if !docpack_path.exists() {
        eprintln!("Error: Docpack file not found: {}", docpack_path.display());
        std::process::exit(1);
    }

    // Check transport mode
    let use_websocket = args.iter().any(|arg| arg == "--websocket" || arg == "-w");
    let use_sse = args.iter().any(|arg| arg == "--sse" || arg == "-s");
    let custom_port = args
        .iter()
        .position(|arg| arg == "--port" || arg == "-p")
        .and_then(|i| args.get(i + 1))
        .and_then(|p| p.parse::<u16>().ok());

    if use_sse {
        // SSE mode (recommended for MCP spec compliance)
        let config = SseConfig {
            host: "0.0.0.0".to_string(),
            port: custom_port.unwrap_or(DEFAULT_MCP_SSE_PORT),
            docpack_path,
        };

        eprintln!("Starting MCP server in SSE mode...");
        serve_sse(config).await?;
    } else if use_websocket {
        // WebSocket mode (for compatibility)
        let config = WebSocketConfig {
            host: "0.0.0.0".to_string(),
            port: custom_port.unwrap_or(DEFAULT_MCP_PORT),
            docpack_path,
        };

        eprintln!("Starting MCP server in WebSocket mode...");
        serve_websocket(config).await?;
    } else {
        // Stdio mode (original behavior)
        eprintln!(
            "Starting MCP server in stdio mode for: {}",
            docpack_path.display()
        );
        let mut server = McpServer::new(docpack_path)?;
        server.run()?;
    }

    Ok(())
}

fn print_usage() {
    eprintln!("Doctown Agent - MCP Server");
    eprintln!();
    eprintln!("USAGE:");
    eprintln!("  doctown-agent <path-to-docpack> [OPTIONS]");
    eprintln!();
    eprintln!("OPTIONS:");
    eprintln!("  --sse, -s           Start in SSE mode (MCP standard, recommended)");
    eprintln!("  --websocket, -w     Start in WebSocket mode");
    eprintln!("  --port, -p <PORT>   Server port (default: 8765)");
    eprintln!();
    eprintln!("MODES:");
    eprintln!("  Stdio mode (default):");
    eprintln!("    Exposes MCP server via JSON-RPC over stdin/stdout");
    eprintln!("    Usage: doctown-agent my-project.docpack");
    eprintln!();
    eprintln!("  SSE mode (recommended for remote access):");
    eprintln!("    Exposes MCP server via HTTP with Server-Sent Events");
    eprintln!("    Follows MCP specification for HTTP transport");
    eprintln!("    Usage: doctown-agent my-project.docpack --sse");
    eprintln!();
    eprintln!("  WebSocket mode:");
    eprintln!("    Exposes MCP server via WebSocket on port 8765");
    eprintln!("    Usage: doctown-agent my-project.docpack --websocket");
    eprintln!();
    eprintln!("EXAMPLES:");
    eprintln!("  # Stdio mode for local agent");
    eprintln!("  doctown-agent ./my-project.docpack");
    eprintln!();
    eprintln!("  # SSE mode on default port (8765) - MCP standard");
    eprintln!("  doctown-agent ./my-project.docpack --sse");
    eprintln!();
    eprintln!("  # WebSocket mode on custom port");
    eprintln!("  doctown-agent ./my-project.docpack -w -p 9000");
    eprintln!();
    eprintln!("DEPLOYMENT:");
    eprintln!("  For RunPod or cloud deployment, use SSE mode:");
    eprintln!("  1. Deploy container with: --network=public");
    eprintln!("  2. Run: doctown-agent /data/project.docpack --sse");
    eprintln!("  3. Connect clients to: http://<runpod-ip>:8765/");
}
