use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::io::Write;
use std::path::Path;
use zip::write::{FileOptions, ZipWriter};
use zip::CompressionMethod;

use crate::ingest::{GraphEdgeKind, ProcessedFile, SymbolNode};

/// Manifest metadata for the .docpack file
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Manifest {
    pub docpack_version: String,
    pub source_repo: Option<String>,
    pub branch: Option<String>,
    pub generated_at: String,
    pub total_files: usize,
    pub total_chunks: usize,
    pub embedding_dimensions: usize,
    pub generator: GeneratorInfo,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GeneratorInfo {
    pub builder_version: String,
    pub embedder: String,
    pub gpu_used: bool,
}

/// File structure representation (hierarchical tree)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FileStructureNode {
    pub path: String,
    pub node_type: String, // "file" or "directory"
    pub size: Option<usize>,
    pub children: Vec<FileStructureNode>,
}

/// Individual chunk representation for chunks.json
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChunkEntry {
    pub chunk_id: String,
    pub file_path: String,
    pub start: usize,
    pub end: usize,
    pub text: String,
}

/// Graph node representation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphNode {
    pub id: String,
    pub node_type: String, // "file", "symbol", "chunk"
    pub name: String,
    pub chunks: Vec<String>,
    pub metadata: HashMap<String, String>,
}

/// Graph edge representation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphEdgeEntry {
    pub from: String,
    pub to: String,
    pub edge_type: String,
    pub score: Option<f32>,
}

/// Complete graph structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphFile {
    pub nodes: Vec<GraphNode>,
    pub edges: Vec<GraphEdgeEntry>,
}

/// Symbol summary for documentation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SymbolSummary {
    pub symbol_id: String,
    pub summary: String,
    pub details: String,
    pub related: Vec<String>,
}

/// Documentation file structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DocumentationFile {
    pub summaries: Vec<SymbolSummary>,
    pub architecture_overview: String,
    pub highlights: Vec<String>,
}

/// Main builder for creating .docpack files
pub struct DocpackBuilder {
    manifest: Manifest,
    file_structure: Vec<FileStructureNode>,
    chunks: Vec<ChunkEntry>,
    embeddings: Vec<crate::ingest::Embedding>,
    graph: GraphFile,
    documentation: DocumentationFile,
}

impl DocpackBuilder {
    pub fn new(source_repo: Option<String>) -> Self {
        let now = chrono::Utc::now().to_rfc3339();
        
        Self {
            manifest: Manifest {
                docpack_version: "7.0.0".to_string(),
                source_repo,
                branch: None,
                generated_at: now,
                total_files: 0,
                total_chunks: 0,
                embedding_dimensions: 0,
                generator: GeneratorInfo {
                    builder_version: "v7".to_string(),
                    embedder: "sentence-transformers/all-MiniLM-L6-v2".to_string(),
                    gpu_used: false, // TODO: detect actual GPU usage
                },
            },
            file_structure: Vec::new(),
            chunks: Vec::new(),
            embeddings: Vec::new(),
            graph: GraphFile {
                nodes: Vec::new(),
                edges: Vec::new(),
            },
            documentation: DocumentationFile {
                summaries: Vec::new(),
                architecture_overview: String::new(),
                highlights: Vec::new(),
            },
        }
    }

    /// Process all files and build internal structures
    pub fn process_files(&mut self, processed_files: Vec<ProcessedFile>) -> Result<()> {
        // Build file structure tree
        self.build_file_structure(&processed_files);

        // Extract all chunks
        self.extract_chunks(&processed_files);

        // Collect all embeddings
        self.collect_embeddings(&processed_files);

        // Build graph
        self.build_graph(&processed_files);

        // Generate documentation
        self.generate_documentation(&processed_files);

        // Update manifest with totals
        self.manifest.total_files = processed_files.len();
        self.manifest.total_chunks = self.chunks.len();
        self.manifest.embedding_dimensions = self.embeddings.first()
            .map(|e| e.len())
            .unwrap_or(0);

        Ok(())
    }

    fn build_file_structure(&mut self, processed_files: &[ProcessedFile]) {
        // Build a flat list for now, could be hierarchical later
        for pf in processed_files {
            self.file_structure.push(FileStructureNode {
                path: pf.file_node.path.clone(),
                node_type: "file".to_string(),
                size: Some(pf.original_bytes.len()),
                children: Vec::new(),
            });
        }
    }

    fn extract_chunks(&mut self, processed_files: &[ProcessedFile]) {
        for pf in processed_files {
            for chunk in &pf.chunks {
                let (start, end) = chunk.byte_range.unwrap_or((0, 0));
                self.chunks.push(ChunkEntry {
                    chunk_id: chunk.id.0.clone(),
                    file_path: pf.file_node.path.clone(),
                    start,
                    end,
                    text: chunk.text.clone(),
                });
            }
        }
    }

    fn collect_embeddings(&mut self, processed_files: &[ProcessedFile]) {
        for pf in processed_files {
            for embedding in &pf.embeddings {
                self.embeddings.push(embedding.clone());
            }
        }
    }

    fn build_graph(&mut self, processed_files: &[ProcessedFile]) {
        // Create file nodes
        for pf in processed_files {
            let mut metadata = HashMap::new();
            metadata.insert("type".to_string(), "file".to_string());
            
            self.graph.nodes.push(GraphNode {
                id: format!("file:{}", pf.file_node.path),
                node_type: "file".to_string(),
                name: pf.file_node.path.clone(),
                chunks: pf.chunks.iter().map(|c| c.id.0.clone()).collect(),
                metadata,
            });
        }

        // Create symbol nodes
        for pf in processed_files {
            for symbol in &pf.symbols {
                let mut metadata = HashMap::new();
                metadata.insert("kind".to_string(), symbol.kind.clone());
                if let Some(vis) = &symbol.visibility {
                    metadata.insert("visibility".to_string(), vis.clone());
                }
                
                self.graph.nodes.push(GraphNode {
                    id: format!("symbol:{}:{}", pf.file_node.path, symbol.name),
                    node_type: "symbol".to_string(),
                    name: symbol.name.clone(),
                    chunks: symbol.chunk_ids.clone(),
                    metadata,
                });
            }
        }

        // Create chunk nodes
        for chunk in &self.chunks {
            self.graph.nodes.push(GraphNode {
                id: chunk.chunk_id.clone(),
                node_type: "chunk".to_string(),
                name: chunk.chunk_id.clone(),
                chunks: vec![chunk.chunk_id.clone()],
                metadata: HashMap::new(),
            });
        }

        // Create edges from all processed files
        for pf in processed_files {
            for edge in &pf.graph_edges {
                let edge_type = match edge.kind {
                    GraphEdgeKind::SymbolToChunk => "symbol_to_chunk",
                    GraphEdgeKind::SymbolToSymbol => "symbol_to_symbol",
                    GraphEdgeKind::FileToFile => "file_to_file",
                    GraphEdgeKind::ChunkToChunk => "chunk_to_chunk",
                };

                self.graph.edges.push(GraphEdgeEntry {
                    from: edge.src.clone(),
                    to: edge.dst.clone(),
                    edge_type: edge_type.to_string(),
                    score: None,
                });
            }
        }

        // --- Semantic: build similarity edges and clusters -----------------
        // We assume `self.embeddings` aligns with `self.chunks` by index.
        let n_chunks = self.embeddings.len();

        if n_chunks > 0 {
            // 1) Build chunk → chunk similarity edges (top-k neighbors)
            let top_k = 5usize.min(n_chunks.saturating_sub(1));
            let sim_threshold = 0.70_f32; // only add edges above this similarity

            for i in 0..n_chunks {
                // Collect similarities to other chunks
                let mut neigh: Vec<(usize, f32)> = Vec::new();
                let a = self.embeddings[i].as_ref();
                for j in 0..n_chunks {
                    if i == j {
                        continue;
                    }
                    let b = self.embeddings[j].as_ref();
                    let score = Self::cosine(a, b);
                    if score >= sim_threshold {
                        neigh.push((j, score));
                    }
                }
                // Keep top_k by score
                neigh.sort_by(|x, y| y.1.partial_cmp(&x.1).unwrap_or(std::cmp::Ordering::Equal));
                neigh.truncate(top_k);

                // Add edges
                for (j, score) in neigh {
                    self.graph.edges.push(GraphEdgeEntry {
                        from: self.chunks[i].chunk_id.clone(),
                        to: self.chunks[j].chunk_id.clone(),
                        edge_type: "semantic_chunk_similarity".to_string(),
                        score: Some(score),
                    });
                }
            }

            // 2) Compute symbol embeddings as mean of their chunk embeddings
            use std::collections::HashMap;
            let mut symbol_embeddings: HashMap<String, Vec<f32>> = HashMap::new();
            let mut symbol_chunk_counts: HashMap<String, usize> = HashMap::new();

            for pf in processed_files {
                for symbol in &pf.symbols {
                    if symbol.chunk_ids.is_empty() {
                        continue;
                    }
                    let mut agg: Vec<f32> = Vec::new();
                    let mut count = 0usize;
                    for cid in &symbol.chunk_ids {
                        if let Some(pos) = self.chunks.iter().position(|c| &c.chunk_id == cid) {
                            let emb = self.embeddings[pos].as_ref();
                            if agg.is_empty() {
                                agg.resize(emb.len(), 0.0);
                            }
                            for (k, &v) in emb.iter().enumerate() {
                                agg[k] += v;
                            }
                            count += 1;
                        }
                    }
                    if count > 0 {
                        for v in agg.iter_mut() {
                            *v /= count as f32;
                        }
                        // normalize symbol embedding
                        Self::normalize_inplace(&mut agg);
                        symbol_embeddings.insert(format!("symbol:{}:{}", pf.file_node.path, symbol.name), agg);
                        symbol_chunk_counts.insert(format!("symbol:{}:{}", pf.file_node.path, symbol.name), count);
                    }
                }
            }

            // 3) Symbol → Symbol similarity (top-3)
            let symbol_ids: Vec<String> = symbol_embeddings.keys().cloned().collect();
            for (idx, sid) in symbol_ids.iter().enumerate() {
                let a = &symbol_embeddings[sid];
                let mut neigh: Vec<(String, f32)> = Vec::new();
                for (jdx, sj) in symbol_ids.iter().enumerate() {
                    if idx == jdx {
                        continue;
                    }
                    let b = &symbol_embeddings[sj];
                    let score = Self::cosine(a.as_slice(), b.as_slice());
                    neigh.push((sj.clone(), score));
                }
                neigh.sort_by(|x, y| y.1.partial_cmp(&x.1).unwrap_or(std::cmp::Ordering::Equal));
                neigh.truncate(3);
                for (other_id, score) in neigh {
                    self.graph.edges.push(GraphEdgeEntry {
                        from: sid.clone(),
                        to: other_id,
                        edge_type: "semantic_symbol_similarity".to_string(),
                        score: Some(score),
                    });
                }
            }

            // 4) Symbol → Chunk top-3 links (by cosine)
            for (sid, emb) in &symbol_embeddings {
                let mut neigh: Vec<(String, f32)> = Vec::new();
                for (i, chunk) in self.chunks.iter().enumerate() {
                    let score = Self::cosine(emb.as_slice(), self.embeddings[i].as_ref());
                    neigh.push((chunk.chunk_id.clone(), score));
                }
                neigh.sort_by(|x, y| y.1.partial_cmp(&x.1).unwrap_or(std::cmp::Ordering::Equal));
                neigh.truncate(3);
                for (cid, score) in neigh {
                    self.graph.edges.push(GraphEdgeEntry {
                        from: sid.clone(),
                        to: cid,
                        edge_type: "semantic_symbol_to_chunk".to_string(),
                        score: Some(score),
                    });
                }
            }

            // 5) Clustering (DBSCAN) on chunk embeddings
            // Use cosine distance: distance = 1 - cosine_similarity (embeddings are normalized)
            let eps = 0.3_f32; // distance threshold (1 - cosine). corresponds to cosine ~0.7
            let min_points = 3usize;
            let assignments = Self::dbscan_cluster(&self.embeddings, eps, min_points);

            // annotate chunks with cluster metadata
            for (i, cluster_opt) in assignments.iter().enumerate() {
                if let Some(node) = self.graph.nodes.iter_mut().find(|n| n.id == self.chunks[i].chunk_id) {
                    match cluster_opt {
                        Some(cid) => {
                            node.metadata.insert("cluster".to_string(), cid.to_string());
                        }
                        None => {
                            node.metadata.insert("cluster".to_string(), "noise".to_string());
                        }
                    }
                }
            }

            // propagate cluster membership to symbols (majority of their chunks, ignoring noise)
            for pf in processed_files {
                for symbol in &pf.symbols {
                    if symbol.chunk_ids.is_empty() {
                        continue;
                    }
                    let mut counts: HashMap<usize, usize> = HashMap::new();
                    let mut total = 0usize;
                    for cid in &symbol.chunk_ids {
                        if let Some(pos) = self.chunks.iter().position(|c| &c.chunk_id == cid) {
                            if let Some(cl) = assignments[pos] {
                                *counts.entry(cl).or_default() += 1;
                                total += 1;
                            }
                        }
                    }
                    if total == 0 {
                        continue;
                    }
                    // pick majority
                    let mut best_cluster: usize = 0;
                    let mut best_count: usize = 0;
                    for (cid, &ct) in counts.iter() {
                        if ct > best_count {
                            best_cluster = *cid;
                            best_count = ct;
                        }
                    }
                    let conf = best_count as f32 / total as f32;
                    let symbol_id = format!("symbol:{}:{}", pf.file_node.path, symbol.name);
                    if let Some(node) = self.graph.nodes.iter_mut().find(|n| n.id == symbol_id) {
                        node.metadata.insert("cluster".to_string(), best_cluster.to_string());
                        node.metadata.insert("cluster_confidence".to_string(), format!("{:.3}", conf));
                    }
                }
            }
        }
    }

    fn generate_documentation(&mut self, processed_files: &[ProcessedFile]) {
        // Rule-based documentation generation (no LLM)
        
        // Generate per-symbol summaries
        for pf in processed_files {
            for symbol in &pf.symbols {
                let summary = self.generate_symbol_summary(symbol);
                
                self.documentation.summaries.push(SymbolSummary {
                    symbol_id: format!("symbol:{}:{}", pf.file_node.path, symbol.name),
                    summary,
                    details: symbol.docs.clone().unwrap_or_default(),
                    related: Vec::new(), // TODO: compute related symbols
                });
            }
        }

        // Generate architecture overview
        self.documentation.architecture_overview = format!(
            "Project contains {} files with {} symbols and {} chunks.",
            processed_files.len(),
            processed_files.iter().map(|pf| pf.symbols.len()).sum::<usize>(),
            self.chunks.len()
        );

        // Generate highlights
        self.documentation.highlights.push(format!(
            "Total files: {}",
            processed_files.len()
        ));
        self.documentation.highlights.push(format!(
            "Total chunks: {}",
            self.chunks.len()
        ));
    }

    fn generate_symbol_summary(&self, symbol: &SymbolNode) -> String {
        let kind_desc = match symbol.kind.as_str() {
            "function" => "Function",
            "struct" => "Struct",
            "enum" => "Enum",
            "trait" => "Trait",
            "impl" => "Implementation",
            "type_alias" => "Type alias",
            "const" => "Constant",
            "static" => "Static variable",
            "mod" => "Module",
            "macro" => "Macro",
            _ => "Symbol",
        };

        let mut summary = format!("{} '{}'", kind_desc, symbol.name);

        if let Some(vis) = &symbol.visibility {
            summary.push_str(&format!(" ({})", vis));
        }

        if let Some(params) = &symbol.parameters {
            if !params.is_empty() {
                summary.push_str(&format!(" with {} parameter(s)", params.len()));
            }
        }

        summary
    }

    // Compute cosine similarity between two slices. Assumes inputs are same length.
    fn cosine(a: &[f32], b: &[f32]) -> f32 {
        let mut dot = 0f32;
        for (x, y) in a.iter().zip(b.iter()) {
            dot += x * y;
        }
        dot
    }

    // Normalize a vector in-place to unit length.
    fn normalize_inplace(v: &mut [f32]) {
        let sum_sq: f32 = v.iter().map(|&x| x * x).sum();
        let norm = sum_sq.sqrt();
        if norm > 1e-12 {
            for x in v.iter_mut() {
                *x /= norm;
            }
        }
    }

    // DBSCAN clustering for normalized embeddings.
    // Returns Vec<Option<cluster_id>> where None indicates noise.
    fn dbscan_cluster(
        embeddings: &Vec<crate::ingest::Embedding>,
        eps: f32,
        min_points: usize,
    ) -> Vec<Option<usize>> {
        // Convert embeddings (Arc<[f32]>) -> Vec<Vec<f32>> (rows)
        let n = embeddings.len();
        let mut labels: Vec<Option<usize>> = vec![None; n];
        if n == 0 {
            return labels;
        }

        let mut inputs: Vec<Vec<f32>> = Vec::with_capacity(n);
        for e in embeddings.iter() {
            inputs.push(e.as_ref().to_vec());
        }

        // Cosine-distance: 1 - cosine_similarity. Embeddings are normalized, so
        // cosine similarity is dot product.
        fn cos_distance(a: &[f32], b: &[f32]) -> f64 {
            let mut dot = 0f64;
            for (x, y) in a.iter().zip(b.iter()) {
                dot += (*x as f64) * (*y as f64);
            }
            (1.0f64 - dot) as f64
        }

        // Use dbscan::Model to cluster with our custom distance
        let model = dbscan::Model::<f32>::new(eps as f64, min_points)
            .set_distance_fn::<fn(&[f32], &[f32]) -> f64>(cos_distance);
        let out = model.run(&inputs);

        // Map dbscan::Classification -> Option<usize>
        for (i, cls) in out.into_iter().enumerate() {
            match cls {
                dbscan::Classification::Core(cid) => labels[i] = Some(cid),
                dbscan::Classification::Edge(cid) => labels[i] = Some(cid),
                dbscan::Classification::Noise => labels[i] = None,
            }
        }

        labels
    }

    /// Write the .docpack file to disk
    pub fn write_to_file<P: AsRef<Path>>(&self, output_path: P) -> Result<()> {
        let file = std::fs::File::create(output_path)?;
        let mut zip = ZipWriter::new(file);
        let options: FileOptions<()> = FileOptions::default()
            .compression_method(CompressionMethod::Deflated)
            .unix_permissions(0o644);

        // Write manifest.json
        zip.start_file("manifest.json", options)?;
        let manifest_json = serde_json::to_string_pretty(&self.manifest)?;
        zip.write_all(manifest_json.as_bytes())?;

        // Write filestructure.json
        zip.start_file("filestructure.json", options)?;
        let filestructure_json = serde_json::to_string_pretty(&self.file_structure)?;
        zip.write_all(filestructure_json.as_bytes())?;

        // Write chunks.json
        zip.start_file("chunks.json", options)?;
        let chunks_json = serde_json::to_string_pretty(&self.chunks)?;
        zip.write_all(chunks_json.as_bytes())?;

        // Write graph.json
        zip.start_file("graph.json", options)?;
        let graph_json = serde_json::to_string_pretty(&self.graph)?;
        zip.write_all(graph_json.as_bytes())?;

        // Write documentation.json
        zip.start_file("documentation.json", options)?;
        let documentation_json = serde_json::to_string_pretty(&self.documentation)?;
        zip.write_all(documentation_json.as_bytes())?;

        // Write embeddings.bin (binary format: row-major f32 array)
        zip.start_file("embeddings.bin", options)?;
        for embedding in &self.embeddings {
            for &value in embedding.as_ref() {
                zip.write_all(&value.to_le_bytes())?;
            }
        }

        // Write README.md
        zip.start_file("README.md", options)?;
        let readme = self.generate_readme();
        zip.write_all(readme.as_bytes())?;

        zip.finish()?;
        Ok(())
    }

    fn generate_readme(&self) -> String {
        format!(
            r#"# Project Documentation (Generated by Doctown v7)

- Source: {}
- Generated: {}
- Files: {}
- Chunks: {}
- Embedding dims: {}

For programmatic use, see manifest.json.

## Structure

- `manifest.json` - Metadata and generation info
- `filestructure.json` - Hierarchical file tree
- `chunks.json` - All text chunks with IDs
- `embeddings.bin` - Binary embedding vectors
- `graph.json` - Semantic project graph
- `documentation.json` - Generated documentation
"#,
            self.manifest.source_repo.as_deref().unwrap_or("(local)"),
            self.manifest.generated_at,
            self.manifest.total_files,
            self.manifest.total_chunks,
            self.manifest.embedding_dimensions
        )
    }
}
