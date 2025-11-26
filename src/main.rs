// main.rs

mod docpack;
mod embedding;
mod ingest;

use anyhow::Result;
use ingest::{code_file_stats, load_zip, unzip_to_memory};

#[tokio::main]
async fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();

    if args.len() < 2 {
        eprintln!("Usage: doctown-builder <zip-path-or-url>");
        std::process::exit(1);
    }

    let source = &args[1];

    // 1: Ingest
    let zip_bytes = load_zip(source).await?;

    // To enable embeddings, create an EmbeddingEngine:
    use embedding::EmbeddingEngine;
    let mut engine = EmbeddingEngine::new(
        "models/minilm-l6/model.onnx",
        "models/minilm-l6/tokenizer.json",
    )?;
    let processed = unzip_to_memory(&zip_bytes, Some(&mut engine)).await?;

    println!("Processed {} files:", processed.len());
    for pf in &processed {
        // Get filename and kind from metadata
        let filename = pf
            .metadata
            .get("path")
            .cloned()
            .or_else(|| pf.metadata.get("filename").cloned())
            .unwrap_or_default();
        let filetype = pf
            .metadata
            .get("filetype")
            .cloned()
            .or_else(|| pf.metadata.get("language").cloned())
            .unwrap_or_default();

        // Filter out entries with missing/unknown filename or filetype
        if filename.is_empty()
            || filetype.is_empty()
            || filename == "<unknown>"
            || filetype == "<unknown>"
        {
            continue;
        }

        // Print filename and file kind
        print!(" - {} [{}]", filename, filetype);

        // Print detailed code stats for code files
        if ["rs", "py", "js", "ts", "cpp", "java"].contains(&filetype.as_str()) {
            if let Some(stats) = code_file_stats(&pf.original_bytes, &filetype) {
                let (total, code, comment, blank) = stats;
                print!(
                    " | lines={} code={} comment={} blank={}",
                    total, code, comment, blank
                );
            }
        }

        // Always print chunk/symbol/embed/meta counts
        println!(
            " | chunks={} symbols={} embeds={} metadata_keys={}",
            pf.chunks.len(),
            pf.symbols.len(),
            pf.embeddings.len(),
            pf.metadata.len()
        );

        // Print detailed symbol breakdown for code files
        if !pf.symbols.is_empty() {
            let mut symbol_counts: std::collections::HashMap<String, usize> =
                std::collections::HashMap::new();
            for sym in &pf.symbols {
                *symbol_counts.entry(sym.kind.clone()).or_insert(0) += 1;
            }
            let mut kinds: Vec<_> = symbol_counts.iter().collect();
            kinds.sort_by_key(|(k, _)| *k);
            print!("   └─ symbols: ");
            for (i, (kind, count)) in kinds.iter().enumerate() {
                if i > 0 {
                    print!(", ");
                }
                print!("{}={}", kind, count);
            }
            println!();
        }

        // Show first few chunk IDs to verify granularity
        if pf.chunks.len() > 0 && filetype == "rs" {
            print!("   └─ chunks: ");
            for (i, chunk) in pf.chunks.iter().take(8).enumerate() {
                if i > 0 {
                    print!(", ");
                }
                // Extract readable name from chunk ID
                let id = &chunk.id.0;
                // Format: "container::method:start-end" or "item::name:start-end"
                let readable = if let Some(double_colon_pos) = id.find("::") {
                    // Has a parent container
                    let after_double = &id[double_colon_pos + 2..];
                    if let Some(single_colon_pos) = after_double.find(':') {
                        &after_double[..single_colon_pos]
                    } else {
                        &id[..double_colon_pos]
                    }
                } else if let Some(single_colon_pos) = id.find(':') {
                    // No parent container
                    &id[..single_colon_pos]
                } else {
                    id
                };
                print!("{}", readable);
            }
            if pf.chunks.len() > 8 {
                print!(" ... (+{})", pf.chunks.len() - 8);
            }
            println!();
        }

        // Print explicit chunk-symbol relationships for code files
        if !pf.chunks.is_empty() && !pf.symbols.is_empty() && filetype == "rs" {
            println!("\n   🔗 CHUNK-SYMBOL RELATIONSHIPS:");

            // Show chunks and their contained symbols
            for chunk in pf.chunks.iter().take(5) {
                let chunk_name = &chunk.id.0;
                println!("      Chunk: {}", chunk_name);
                if chunk.containing_symbols.is_empty() {
                    println!("        └─ (no symbols in this chunk)");
                } else {
                    println!(
                        "        └─ contains {} symbol(s):",
                        chunk.containing_symbols.len()
                    );
                    for (i, sym_name) in chunk.containing_symbols.iter().enumerate() {
                        if i < 5 {
                            // Find symbol details
                            if let Some(sym) = pf.symbols.iter().find(|s| &s.name == sym_name) {
                                println!("           • {} ({})", sym.name, sym.kind);
                            }
                        }
                    }
                    if chunk.containing_symbols.len() > 5 {
                        println!(
                            "           ... and {} more",
                            chunk.containing_symbols.len() - 5
                        );
                    }
                }
            }
            if pf.chunks.len() > 5 {
                println!("      ... and {} more chunks\n", pf.chunks.len() - 5);
            }

            // Show symbols and which chunks they're in
            println!("   🔗 SYMBOL-CHUNK RELATIONSHIPS:");
            for symbol in pf.symbols.iter().take(5) {
                println!("      Symbol: {} ({})", symbol.name, symbol.kind);
                if symbol.chunk_ids.is_empty() {
                    println!("        └─ (not in any chunk)");
                } else {
                    println!("        └─ lives in {} chunk(s):", symbol.chunk_ids.len());
                    for (i, chunk_id) in symbol.chunk_ids.iter().enumerate() {
                        if i < 3 {
                            println!("           • {}", chunk_id);
                        }
                    }
                    if symbol.chunk_ids.len() > 3 {
                        println!("           ... and {} more", symbol.chunk_ids.len() - 3);
                    }
                }
            }
            if pf.symbols.len() > 5 {
                println!("      ... and {} more symbols\n", pf.symbols.len() - 5);
            }
        }
    }

    // Build the semantic project graph
    println!("\n=== Building Semantic Project Graph ===");
    let graph = ingest::ProjectGraph::from_processed_files(processed.clone());
    graph.print_summary();

    // Generate .docpack file
    println!("\n=== Generating .docpack file ===");
    let mut builder = docpack::DocpackBuilder::new(Some(source.to_string()));
    builder.process_files(processed)?;

    // Determine output filename - always use simple name in current directory
    let output_path = if source.contains("github.com") {
        // Extract repo name from URL like: https://github.com/user/repo/archive/refs/heads/main.zip
        let parts: Vec<&str> = source.split('/').collect();
        // Find "github.com" and get the repo name (2 positions after)
        if let Some(pos) = parts.iter().position(|&p| p == "github.com") {
            if pos + 2 < parts.len() {
                format!("{}.docpack", parts[pos + 2])
            } else {
                "output.docpack".to_string()
            }
        } else {
            "output.docpack".to_string()
        }
    } else if source.ends_with(".zip") {
        // Local file - just replace extension
        let path = std::path::Path::new(source);
        let filename = path.file_stem().unwrap_or_default().to_string_lossy();
        format!("{}.docpack", filename)
    } else {
        "output.docpack".to_string()
    };

    builder.write_to_file(&output_path)?;
    println!("✅ Generated: {}", output_path);

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_localdoc_repo() {
        let result =
            load_zip("https://github.com/xandwr/localdoc/archive/refs/heads/main.zip").await;
        assert!(result.is_ok());
    }
}
