// main.rs

mod community;
mod docgen;
mod docpack;
mod embedding;
mod ingest;

use anyhow::Result;
use ingest::{code_file_stats, load_zip, unzip_to_memory_parallel};
use chrono::{DateTime, Utc};
use std::time::Instant;

/// Lightweight runtime execution report collector for `main`.
struct ExecutionReport {
    wall_start: Option<DateTime<Utc>>,
    args_parsed: Option<DateTime<Utc>>,
    before_zip_load: Option<DateTime<Utc>>,
    zip_loaded: Option<DateTime<Utc>>,
    engine_initialized: Option<DateTime<Utc>>,
    before_processing: Option<DateTime<Utc>>,
    processed: Option<DateTime<Utc>>,
    graph_built: Option<DateTime<Utc>>,
    docpack_written: Option<DateTime<Utc>>,
    finished: Option<DateTime<Utc>>,
    total_wall_seconds: Option<f64>,
}

impl ExecutionReport {
    fn new() -> Self {
        Self {
            wall_start: None,
            args_parsed: None,
            before_zip_load: None,
            zip_loaded: None,
            engine_initialized: None,
            before_processing: None,
            processed: None,
            graph_built: None,
            docpack_written: None,
            finished: None,
            total_wall_seconds: None,
        }
    }

    fn print(&self) {
        println!("\n=== Execution Report ===");
        let mut prev: Option<DateTime<Utc>> = self.wall_start.or(self.args_parsed);

        macro_rules! show {
            ($label:expr, $t:expr) => {
                if let Some(ts) = $t {
                    let s = ts.to_rfc3339();
                    let delta = if let Some(p) = prev {
                        let d = ts.signed_duration_since(p);
                        format!("+{:.3}s", d.num_milliseconds() as f64 / 1000.0)
                    } else {
                        "".to_string()
                    };
                    println!(" - {:20}: {} {}", $label, s, delta);
                    prev = Some(ts);
                }
            };
        }

        show!("wall_start", self.wall_start);
        show!("args_parsed", self.args_parsed);
        show!("before_zip_load", self.before_zip_load);
        show!("zip_loaded", self.zip_loaded);
        show!("engine_init", self.engine_initialized);
        show!("before_processing", self.before_processing);
        show!("processed", self.processed);
        show!("graph_built", self.graph_built);
        show!("docpack_written", self.docpack_written);
        show!("finished", self.finished);

        if let Some(total) = self.total_wall_seconds {
            println!(" - {:20}: total_wall_secs = {:.3}s", "summary", total);
        }
        println!("========================\n");
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    let wall_start: DateTime<Utc> = Utc::now();
    let run_start = Instant::now();

    let mut report = ExecutionReport::new();

    let args: Vec<String> = std::env::args().collect();
    report.args_parsed = Some(Utc::now());

    if args.len() < 2 {
        eprintln!("Usage: doctown-builder <zip-path-or-url>");
        std::process::exit(1);
    }

    let source = &args[1];

    // 1: Ingest
    report.before_zip_load = Some(Utc::now());
    let zip_bytes = load_zip(source).await?;
    report.zip_loaded = Some(Utc::now());

    // To enable embeddings, create an EmbeddingEngine:
    use embedding::EmbeddingEngine;
    let mut engine = EmbeddingEngine::new(
        "models/minilm-l6/model.onnx",
        "models/minilm-l6/tokenizer.json",
    )?;
    report.engine_initialized = Some(Utc::now());

    // Use the parallel pipeline for maximum performance:
    // - Parses & chunks files in parallel with Rayon
    // - Batches all embeddings together for GPU efficiency
    // - On 4070 Ti: 5,000 chunks embed in ~150ms
    println!("🚀 Using parallel pipeline...");
    report.before_processing = Some(Utc::now());
    let start = Instant::now();
    let processed = unzip_to_memory_parallel(&zip_bytes, Some(&mut engine)).await?;
    let elapsed = start.elapsed();
    report.processed = Some(Utc::now());
    println!("✅ Processed in {:.2}s", elapsed.as_secs_f64());

    // For comparison, the old sequential version:
    // let processed = unzip_to_memory(&zip_bytes, Some(&mut engine)).await?;

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
            // Prefer original bytes when available; otherwise fall back to stored metadata
            let stats_opt = if !pf.original_bytes.is_empty() {
                code_file_stats(&pf.original_bytes, &filetype)
            } else if let Some(total_s) = pf.metadata.get("loc_total") {
                // Parse stored metadata fields inserted during ingestion
                let total = total_s.parse::<usize>().unwrap_or(0);
                let code = pf
                    .metadata
                    .get("loc_code")
                    .and_then(|s| s.parse::<usize>().ok())
                    .unwrap_or(0);
                let comment = pf
                    .metadata
                    .get("loc_comment")
                    .and_then(|s| s.parse::<usize>().ok())
                    .unwrap_or(0);
                let blank = pf
                    .metadata
                    .get("loc_blank")
                    .and_then(|s| s.parse::<usize>().ok())
                    .unwrap_or(0);
                Some((total, code, comment, blank))
            } else {
                None
            };

            if let Some((total, code, comment, blank)) = stats_opt {
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

    // Build the semantic project graph with similarity edges and community detection
    println!("\n=== Building Semantic Project Graph ===");
    let graph = ingest::ProjectGraph::from_processed_files_with_communities(
        processed.clone(),
        0.5, // similarity threshold: only connect chunks with cosine similarity >= 0.5
    );
    graph.print_summary();

    // Print detected communities/subsystems and refactor suggestions
    graph.print_communities();
    report.graph_built = Some(Utc::now());

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

    report.docpack_written = Some(Utc::now());
    report.finished = Some(Utc::now());
    report.wall_start = Some(wall_start);
    report.total_wall_seconds = Some(run_start.elapsed().as_secs_f64());

    // Print concise execution report
    report.print();

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
