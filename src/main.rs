// main.rs

mod agent;
mod community;
mod nlp;
mod docgen;
mod docpack;
mod embedding;
mod ingest;

use anyhow::Result;
use ingest::{code_file_stats, load_zip, unzip_to_memory_parallel};
use std::collections::{HashMap, HashSet};
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

        // Show condensed semantic blocks (group consecutive chunks that share >=60% symbols)
        if pf.chunks.len() > 0 && filetype == "rs" {
            // Build symbol sets per chunk
            let mut chunk_symbol_sets: Vec<HashSet<String>> = Vec::new();
            for chunk in &pf.chunks {
                let set: HashSet<String> = chunk.containing_symbols.iter().cloned().collect();
                chunk_symbol_sets.push(set);
            }

            // Threshold for sharing (60%)
            let threshold: f32 = 0.6;

            // Greedy grouping of consecutive chunks where adjacent chunks share >=60% of the smaller chunk's symbols
            let mut blocks: Vec<Vec<usize>> = Vec::new();
            let mut i = 0usize;
            while i < chunk_symbol_sets.len() {
                let mut block = vec![i];
                let mut j = i + 1;
                while j < chunk_symbol_sets.len() {
                    let a = &chunk_symbol_sets[*block.last().unwrap()];
                    let b = &chunk_symbol_sets[j];

                    let inter: HashSet<_> = a.intersection(b).collect();
                    let min_size = usize::min(a.len(), b.len());

                    let share = if min_size == 0 { 0.0 } else { inter.len() as f32 / min_size as f32 };

                    if share >= threshold {
                        block.push(j);
                        j += 1;
                    } else {
                        break;
                    }
                }
                blocks.push(block);
                i = blocks.last().unwrap().last().unwrap().saturating_add(1);
            }

            // Print narrative summary of semantic blocks
            println!("   └─ Semantic blocks: {}", blocks.len());
            for (bi, block) in blocks.iter().enumerate() {
                // Representative chunk name (first chunk readable id)
                let first_chunk = &pf.chunks[block[0]].id.0;
                let repr_name = if let Some(pos) = first_chunk.find("::") {
                    let after = &first_chunk[pos + 2..];
                    if let Some(colpos) = after.find(':') {
                        &after[..colpos]
                    } else {
                        &first_chunk[..pos]
                    }
                } else if let Some(pos) = first_chunk.find(':') {
                    &first_chunk[..pos]
                } else {
                    &first_chunk
                };

                // Union of symbols in block
                let mut union_syms: HashSet<String> = HashSet::new();
                for &ci in block {
                    for s in &pf.chunks[ci].containing_symbols {
                        union_syms.insert(s.clone());
                    }
                }

                // Representative symbols (top by frequency)
                let mut freq: HashMap<String, usize> = HashMap::new();
                for &ci in block {
                    for s in &pf.chunks[ci].containing_symbols {
                        *freq.entry(s.clone()).or_default() += 1;
                    }
                }
                let mut freq_vec: Vec<_> = freq.into_iter().collect();
                freq_vec.sort_by(|a, b| b.1.cmp(&a.1));
                let repr_syms: Vec<String> = freq_vec.iter().take(6).map(|(s, _)| s.clone()).collect();

                println!("     {}. Unit ({} chunk(s)) — repr: {}", bi + 1, block.len(), repr_name);
                if !repr_syms.is_empty() {
                    println!("        └─ composed of symbols: {}", repr_syms.join(", "));
                } else if !union_syms.is_empty() {
                    let mut u: Vec<_> = union_syms.into_iter().collect();
                    u.truncate(6);
                    println!("        └─ composed of symbols: {}", u.join(", "));
                } else {
                    println!("        └─ (no recognized symbols in this unit)");
                }

                // Symbol relationships: summarize which symbols appear in which chunks and any local symbol-to-symbol edges
                println!("        └─ symbol relations:");
                // Build a quick map of symbol -> chunk indices within this block
                let mut sym_to_chunks: HashMap<String, Vec<usize>> = HashMap::new();
                for &ci in block {
                    for s in &pf.chunks[ci].containing_symbols {
                        sym_to_chunks.entry(s.clone()).or_default().push(ci);
                    }
                }

                // Show up to 8 symbols
                let mut shown = 0usize;
                for (sym, locations) in sym_to_chunks.iter() {
                    if shown >= 8 { break; }
                    shown += 1;
                    let locs_str: Vec<String> = locations.iter().map(|idx| pf.chunks[*idx].id.0.clone()).collect();
                    print!("           • {} — in {} chunk(s)", sym, locations.len());
                    if !locs_str.is_empty() {
                        print!(": {}", locs_str.into_iter().take(3).collect::<Vec<_>>().join(", "));
                        if locations.len() > 3 { print!(" (+{} more)", locations.len() - 3); }
                    }
                    println!("");
                }
                if sym_to_chunks.len() > 8 {
                    println!("           ... and {} more symbols", sym_to_chunks.len() - 8);
                }
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

    // Recompute chunk -> community mapping so we can merge semantic blocks by cluster
    // Build similarity edges and run Louvain locally to get chunk indices -> community id
    {
        use community::{build_similarity_edges, Louvain};
        let n_chunks = graph.chunk_embeddings.len();
        if n_chunks > 0 {
            let sim_edges = build_similarity_edges(&graph.chunk_embeddings, 0.5);
            let mut l = Louvain::new(n_chunks, &sim_edges);
            let res = l.run();

            // Map chunk id string -> global index
            let mut chunkid_to_index: HashMap<String, usize> = HashMap::new();
            for (idx, c) in graph.chunks.iter().enumerate() {
                chunkid_to_index.insert(c.id.0.clone(), idx);
            }

            // Build chunk index -> community id map
            let mut chunk_to_comm: HashMap<usize, usize> = HashMap::new();
            for (comm_id, indices) in &res.communities {
                for &ci in indices {
                    chunk_to_comm.insert(ci, *comm_id);
                }
            }

            // Now for each processed file, recompute semantic blocks and merge adjacent ones
            println!("\n=== File-level Semantic Blocks (merged) ===");
            for pf in &processed {
                let file_path = &pf.file_node.path;
                // Build symbol name -> kind map for this file
                let mut sym_kind: HashMap<String, String> = HashMap::new();
                for s in &pf.symbols {
                    sym_kind.insert(s.name.clone(), s.kind.clone());
                }

                if pf.chunks.is_empty() {
                    continue;
                }

                // chunk symbol sets and kind sets and embeddings per chunk
                let mut chunk_symbol_sets: Vec<HashSet<String>> = Vec::new();
                let mut chunk_kind_sets: Vec<HashSet<String>> = Vec::new();
                for (_ci, chunk) in pf.chunks.iter().enumerate() {
                    let mut ks: HashSet<String> = HashSet::new();
                    for s in &chunk.containing_symbols {
                        if let Some(k) = sym_kind.get(s) {
                            ks.insert(k.clone());
                        }
                    }
                    let sset: HashSet<String> = chunk.containing_symbols.iter().cloned().collect();
                    chunk_symbol_sets.push(sset);
                    chunk_kind_sets.push(ks);
                }

                // initial greedy blocks (consecutive chunks sharing >=60% of smaller chunk symbols)
                let mut blocks: Vec<Vec<usize>> = Vec::new();
                let mut i = 0usize;
                while i < chunk_symbol_sets.len() {
                    let mut block = vec![i];
                    let mut j = i + 1;
                    while j < chunk_symbol_sets.len() {
                        let a = &chunk_symbol_sets[*block.last().unwrap()];
                        let b = &chunk_symbol_sets[j];
                        let inter: HashSet<_> = a.intersection(b).collect();
                        let min_size = usize::min(a.len(), b.len());
                        let share = if min_size == 0 { 0.0 } else { inter.len() as f32 / min_size as f32 };
                        if share >= 0.6 {
                            block.push(j);
                            j += 1;
                        } else {
                            break;
                        }
                    }
                    blocks.push(block);
                    i = blocks.last().unwrap().last().unwrap().saturating_add(1);
                }

                // Helper to compute union symbol set for a block
                let union_symbols = |block: &Vec<usize>| -> HashSet<String> {
                    let mut u = HashSet::new();
                    for &ci in block {
                        for s in &pf.chunks[ci].containing_symbols {
                            u.insert(s.clone());
                        }
                    }
                    u
                };

                // Helper to compute union kind set for a block
                let union_kinds = |block: &Vec<usize>| -> HashSet<String> {
                    let mut u = HashSet::new();
                    for &ci in block {
                        for k in &chunk_kind_sets[ci] {
                            u.insert(k.clone());
                        }
                    }
                    u
                };

                // Helper to compute centroid embedding for a block (if embeddings exist)
                let block_centroid = |block: &Vec<usize>| -> Option<Vec<f32>> {
                    if pf.embeddings.is_empty() {
                        return None;
                    }
                    let dim = pf.embeddings[0].len();
                    let mut cent = vec![0.0f32; dim];
                    let mut cnt = 0usize;
                    for &ci in block {
                        if ci < pf.embeddings.len() {
                            for (d, v) in cent.iter_mut().zip(pf.embeddings[ci].iter()) {
                                *d += *v;
                            }
                            cnt += 1;
                        }
                    }
                    if cnt == 0 { return None; }
                    for d in cent.iter_mut() { *d /= cnt as f32; }
                    Some(cent)
                };

                // Merge adjacent blocks based on rules repeatedly until stable
                let mut merged = true;
                while merged {
                    merged = false;
                    let mut new_blocks: Vec<Vec<usize>> = Vec::new();
                    let mut idx = 0usize;
                    while idx < blocks.len() {
                        if idx + 1 >= blocks.len() {
                            new_blocks.push(blocks[idx].clone());
                            break;
                        }

                        let left = &blocks[idx];
                        let right = &blocks[idx + 1];

                        // Condition A: belong to same cluster (if available)
                        let mut same_cluster = false;
                        // Get any global chunk indices for left/right blocks
                        let mut left_global: Vec<usize> = Vec::new();
                        let mut right_global: Vec<usize> = Vec::new();
                        for &ci in left {
                            if let Some(global_idx) = chunkid_to_index.get(&pf.chunks[ci].id.0) {
                                left_global.push(*global_idx);
                            }
                        }
                        for &ci in right {
                            if let Some(global_idx) = chunkid_to_index.get(&pf.chunks[ci].id.0) {
                                right_global.push(*global_idx);
                            }
                        }
                        if !left_global.is_empty() && !right_global.is_empty() {
                            // If any pair share same community id, treat as same cluster
                            'outer: for &lg in &left_global {
                                if let Some(lc) = chunk_to_comm.get(&lg) {
                                    for &rg in &right_global {
                                        if let Some(rc) = chunk_to_comm.get(&rg) {
                                            if lc == rc {
                                                same_cluster = true;
                                                break 'outer;
                                            }
                                        }
                                    }
                                }
                            }
                        }

                        // Condition B: share >=50% of symbols between unions
                        let u_left = union_symbols(left);
                        let u_right = union_symbols(right);
                        let inter_count = u_left.intersection(&u_right).count();
                        let min_sz = usize::min(u_left.len(), u_right.len());
                        let share = if min_sz == 0 { 0.0 } else { inter_count as f32 / min_sz as f32 };

                        // Asymmetric thresholds depending on scope
                        // Determine if chunks share same parent container (e.g., same function/impl)
                        let left_first = &pf.chunks[left[0]].id.0;
                        let right_first = &pf.chunks[right[0]].id.0;
                        let left_parent = left_first.split("::").next().unwrap_or("");
                        let right_parent = right_first.split("::").next().unwrap_or("");
                        let same_parent = left_parent == right_parent && !left_parent.is_empty();

                        let threshold = if same_parent {
                            0.25 // intra-function/impl
                        } else {
                            // same file (we're iterating within a file) -> 0.35
                            0.35
                        };

                        let cross_file_threshold = 0.55f32;

                        // Condition: centroid similarity >= threshold depending on scope
                        let mut centroid_sim_ok = false;
                        if let (Some(lc), Some(rc)) = (block_centroid(left), block_centroid(right)) {
                            let sim = community::cosine_similarity(&lc, &rc);
                            // if chunks map to global indices in different files, use cross-file threshold
                            let cross_file = left_global.iter().any(|lg| {
                                right_global.iter().any(|rg| {
                                    graph.chunk_to_file.get(lg) != graph.chunk_to_file.get(rg)
                                })
                            });
                            let use_thresh = if cross_file { cross_file_threshold } else { threshold };
                            if sim >= use_thresh {
                                centroid_sim_ok = true;
                            }
                        }

                        // Condition D: symbol kinds identical (primary kinds equal)
                        let lk = union_kinds(left);
                        let rk = union_kinds(right);
                        let kinds_identical = !lk.is_empty() && lk == rk;

                        // Structural link: any explicit chunk->chunk edge between globals
                        let mut structural_link = false;
                        'edgecheck: for &lg in &left_global {
                            for &rg in &right_global {
                                for e in &graph.edges {
                                    if (e.src == graph.chunks[lg].id.0 && e.dst == graph.chunks[rg].id.0)
                                        || (e.dst == graph.chunks[lg].id.0 && e.src == graph.chunks[rg].id.0)
                                    {
                                        if matches!(e.kind, ingest::GraphEdgeKind::ChunkToChunk | ingest::GraphEdgeKind::SymbolSimilarity) {
                                            structural_link = true;
                                            break 'edgecheck;
                                        }
                                    }
                                }
                            }
                        }

                        // Never merge across top-level structural boundaries (mod, enum, function, struct)
                        let forbid_merge_due_to_boundary = {
                            let left_kind = left_first.split("::").next().unwrap_or("");
                            let right_kind = right_first.split("::").next().unwrap_or("");
                            let top_kinds = ["mod", "struct", "enum", "fn", "trait", "impl"];
                            top_kinds.contains(&left_kind) && top_kinds.contains(&right_kind) && left_kind != right_kind
                        };

                        // Merge allowed only if one of the conditions true AND (>=3 shared symbols or structural link)
                        let raw_condition = same_cluster || (share >= 0.5) || centroid_sim_ok || kinds_identical;
                        let share_count_ok = inter_count >= 3;

                        if !forbid_merge_due_to_boundary && raw_condition && (share_count_ok || structural_link) {
                            // Merge into one block
                            let mut merged_block = left.clone();
                            merged_block.extend(right.iter());
                            new_blocks.push(merged_block);
                            idx += 2;
                            merged = true;
                        } else {
                            new_blocks.push(blocks[idx].clone());
                            idx += 1;
                        }
                    }
                    blocks = new_blocks;
                }

                // Print merged blocks for this file (only if more than 0)
                println!(" - {} ({} chunk(s))", file_path, pf.chunks.len());
                println!("   └─ semantic units: {}", blocks.len());
                for (bi, block) in blocks.iter().enumerate() {
                    // representative name
                    let first_chunk = &pf.chunks[block[0]].id.0;
                    let repr_name = if let Some(pos) = first_chunk.find("::") {
                        let after = &first_chunk[pos + 2..];
                        if let Some(colpos) = after.find(':') { &after[..colpos] } else { &first_chunk[..pos] }
                    } else if let Some(pos) = first_chunk.find(':') { &first_chunk[..pos] } else { &first_chunk };

                    // representative symbols
                    let mut freq: HashMap<String, usize> = HashMap::new();
                    for &ci in block {
                        for s in &pf.chunks[ci].containing_symbols {
                            *freq.entry(s.clone()).or_default() += 1;
                        }
                    }
                    let mut freq_vec: Vec<_> = freq.into_iter().collect();
                    freq_vec.sort_by(|a, b| b.1.cmp(&a.1));
                    let repr_syms: Vec<String> = freq_vec.iter().take(6).map(|(s, _)| s.clone()).collect();

                    println!("     {}. Unit ({} chunk(s)) — repr: {}", bi + 1, block.len(), repr_name);
                    if !repr_syms.is_empty() {
                        println!("        └─ composed of symbols: {}", repr_syms.join(", "));
                    } else {
                        println!("        └─ (no recognized symbols in this unit)");
                    }
                }
            }
        }
    }

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

    builder.write_to_file(&output_path, Some(&graph))?;
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
