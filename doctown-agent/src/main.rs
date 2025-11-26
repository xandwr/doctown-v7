// main.rs - Doctown Agent MCP server entry point

use anyhow::Result;
use doctown_agent::{DEFAULT_MCP_PORT, McpServer, WebSocketConfig, serve_websocket};
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

    // Check if WebSocket mode is requested
    let use_websocket = args.iter().any(|arg| arg == "--websocket" || arg == "-w");
    let custom_port = args
        .iter()
        .position(|arg| arg == "--port" || arg == "-p")
        .and_then(|i| args.get(i + 1))
        .and_then(|p| p.parse::<u16>().ok());

    if use_websocket {
        // WebSocket mode
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
    eprintln!("  --websocket, -w     Start in WebSocket mode (default: stdio)");
    eprintln!("  --port, -p <PORT>   WebSocket port (default: 8765)");
    eprintln!();
    eprintln!("MODES:");
    eprintln!("  Stdio mode (default):");
    eprintln!("    Exposes MCP server via JSON-RPC over stdin/stdout");
    eprintln!("    Usage: doctown-agent my-project.docpack");
    eprintln!();
    eprintln!("  WebSocket mode:");
    eprintln!("    Exposes MCP server via WebSocket on port 8765 (or custom port)");
    eprintln!("    Usage: doctown-agent my-project.docpack --websocket");
    eprintln!("    Usage: doctown-agent my-project.docpack -w --port 9000");
    eprintln!();
    eprintln!("EXAMPLES:");
    eprintln!("  # Stdio mode for local agent");
    eprintln!("  doctown-agent ./my-project.docpack");
    eprintln!();
    eprintln!("  # WebSocket mode on default port (8765)");
    eprintln!("  doctown-agent ./my-project.docpack --websocket");
    eprintln!();
    eprintln!("  # WebSocket mode on custom port");
    eprintln!("  doctown-agent ./my-project.docpack -w -p 9000");
    eprintln!();
    eprintln!("DEPLOYMENT:");
    eprintln!("  For RunPod or cloud deployment, use WebSocket mode:");
    eprintln!("  1. Deploy container with: --network=public");
    eprintln!("  2. Run: doctown-agent /data/project.docpack --websocket");
    eprintln!("  3. Connect clients to: ws://<runpod-ip>:8765/");
}
