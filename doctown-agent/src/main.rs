// main.rs - Doctown Agent MCP server entry point

use anyhow::Result;
use doctown_agent::McpServer;
use std::path::PathBuf;

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();

    if args.len() < 2 {
        eprintln!("Usage: doctown-agent <path-to-docpack>");
        eprintln!();
        eprintln!(
            "Starts an MCP server that exposes the docpack contents via JSON-RPC over stdio."
        );
        std::process::exit(1);
    }

    let docpack_path = PathBuf::from(&args[1]);

    if !docpack_path.exists() {
        eprintln!("Error: Docpack file not found: {}", docpack_path.display());
        std::process::exit(1);
    }

    eprintln!("Starting MCP server for: {}", docpack_path.display());

    let mut server = McpServer::new(docpack_path)?;
    server.run()?;

    Ok(())
}
