use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::io::Write;
use std::path::Path;
use zip::CompressionMethod;
use zip::write::{FileOptions, ZipWriter};

use crate::docgen::DocGenerator;
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
    #[serde(skip_serializing_if = "Option::is_none")]
    pub file_hashes: Option<HashMap<String, String>>, // file_path -> SHA256 hash
    #[serde(skip_serializing_if = "Option::is_none")]
    pub git_commit: Option<String>, // Git commit hash when docpack was built
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

/// Documentation file structure (enhanced with comprehensive docs)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DocumentationFile {
    pub summaries: Vec<SymbolSummary>,
    pub architecture_overview: String,
    pub highlights: Vec<String>,
    // Enhanced documentation from docgen
    pub module_summaries: Vec<ModuleSummaryData>,
    pub file_summaries: Vec<FileSummaryData>,
    pub struct_docs: Vec<StructDocData>,
    pub function_docs: Vec<FunctionDocData>,
    pub dependency_overview: DependencyOverviewData,
    pub cluster_summaries: Vec<ClusterSummaryData>,
}

// Serializable versions of docgen structures
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModuleSummaryData {
    pub module_path: String,
    pub description: String,
    pub file_count: usize,
    pub symbol_count: usize,
    pub lines_of_code: usize,
    pub primary_purpose: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FileSummaryData {
    pub file_path: String,
    pub description: String,
    pub language: String,
    pub lines_total: usize,
    pub lines_code: usize,
    pub complexity_score: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StructDocData {
    pub name: String,
    pub file_path: String,
    pub description: String,
    pub visibility: String,
    pub field_count: usize,
    pub method_count: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FunctionDocData {
    pub name: String,
    pub file_path: String,
    pub description: String,
    pub signature: String,
    pub complexity_estimate: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DependencyOverviewData {
    pub internal_count: usize,
    pub external_count: usize,
    pub circular_count: usize,
    pub summary: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClusterSummaryData {
    pub cluster_id: String,
    pub topic_label: String,
    pub description: String,
    pub symbol_count: usize,
}

/// Agent-optimized index for LLM navigation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentIndex {
    pub version: String,
    pub subsystems: Vec<Subsystem>,
    pub symbols: HashMap<String, SymbolEntry>,
    pub tasks: HashMap<String, Vec<String>>,
    pub impact_graph: HashMap<String, Vec<String>>,
    pub quickstart: AgentQuickstart,
}

/// A subsystem (community) of related code
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Subsystem {
    pub name: String,
    pub symbols: Vec<String>,
    pub files: Vec<String>,
    pub confidence: f32,
    pub role: String,
    pub summary: String,
}

/// Comprehensive symbol entry for agent navigation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SymbolEntry {
    pub kind: String,
    pub file: String,
    pub chunk: String, // "start_byte-end_byte" or "line_start-line_end"
    pub subsystem: String,
    pub signature: Option<String>,
    pub summary: String,
    pub related_symbols: Vec<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub embedding_index: Option<usize>, // Index into embeddings array for semantic search
    #[serde(skip_serializing_if = "Option::is_none")]
    pub used_in_tests: Option<Vec<String>>, // Test files that reference this symbol
    #[serde(skip_serializing_if = "Option::is_none")]
    pub mentioned_in_docs: Option<Vec<String>>, // Documentation files that mention this symbol
    #[serde(skip_serializing_if = "Option::is_none")]
    pub usage_examples: Option<Vec<UsageExample>>, // Actual code examples showing how this symbol is used
    #[serde(skip_serializing_if = "Option::is_none")]
    pub llm_summary: Option<String>, // LLM-generated plain-English "Explain Like I'm 5" summary
}

/// A concrete usage example of a symbol
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UsageExample {
    pub file: String,          // File where the example is found
    pub context: String,       // Surrounding code showing how the symbol is used
    pub example_type: String,  // "test", "doc", "usage"
    pub line_start: usize,     // Starting line number
    pub line_end: usize,       // Ending line number
}

/// Quick-start navigation hints for agents
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentQuickstart {
    pub entry_points: Vec<String>,
    pub core_types: Vec<String>,
    pub most_connected: Vec<String>,
    pub subsystem_map: HashMap<String, String>,
}

/// Main builder for creating .docpack files
pub struct DocpackBuilder {
    manifest: Manifest,
    file_structure: Vec<FileStructureNode>,
    chunks: Vec<ChunkEntry>,
    embeddings: Vec<crate::ingest::Embedding>,
    graph: GraphFile,
    documentation: DocumentationFile,
    processed_files: Vec<crate::ingest::ProcessedFile>,
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
                file_hashes: None,
                git_commit: None,
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
                module_summaries: Vec::new(),
                file_summaries: Vec::new(),
                struct_docs: Vec::new(),
                function_docs: Vec::new(),
                dependency_overview: DependencyOverviewData {
                    internal_count: 0,
                    external_count: 0,
                    circular_count: 0,
                    summary: String::new(),
                },
                cluster_summaries: Vec::new(),
            },
            processed_files: Vec::new(),
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
        self.manifest.embedding_dimensions = self.embeddings.first().map(|e| e.len()).unwrap_or(0);

        // Store processed files for later writing to zip
        self.processed_files = processed_files;

        Ok(())
    }

    /// Compute SHA256 hashes for all processed files
    pub fn compute_file_hashes(&mut self) {
        use sha2::{Sha256, Digest};
        let mut hashes = HashMap::new();

        for pf in &self.processed_files {
            let mut hasher = Sha256::new();
            hasher.update(&pf.original_bytes);
            let hash = format!("{:x}", hasher.finalize());
            hashes.insert(pf.file_node.path.clone(), hash);
        }

        self.manifest.file_hashes = Some(hashes);
    }

    /// Detect and store the current git commit hash
    pub fn detect_git_commit(&mut self) {
        // Try to get the current git commit using git command
        if let Ok(output) = std::process::Command::new("git")
            .args(&["rev-parse", "HEAD"])
            .output()
        {
            if output.status.success() {
                if let Ok(commit) = String::from_utf8(output.stdout) {
                    self.manifest.git_commit = Some(commit.trim().to_string());
                }
            }
        }
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
                    GraphEdgeKind::SymbolSimilarity => "symbol_similarity",
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
            println!(
                "⚡ Computing semantic similarities for {} chunks...",
                n_chunks
            );

            // Build a chunk_id -> index map for fast lookups
            use std::collections::HashMap;
            let chunk_id_to_idx: HashMap<String, usize> = self
                .chunks
                .iter()
                .enumerate()
                .map(|(i, c)| (c.chunk_id.clone(), i))
                .collect();

            // 1) Build chunk → chunk similarity edges (top-k neighbors)
            let top_k = 5usize.min(n_chunks.saturating_sub(1));
            let sim_threshold = 0.70_f32; // only add edges above this similarity

            for i in 0..n_chunks {
                // Collect similarities to other chunks
                let mut neigh: Vec<(usize, f32)> = Vec::new();
                let a = self.embeddings[i].as_ref();
                for j in (i + 1)..n_chunks {
                    // Only compute upper triangle, add both directions
                    let b = self.embeddings[j].as_ref();
                    let score = Self::cosine(a, b);
                    if score >= sim_threshold {
                        neigh.push((j, score));
                    }
                }
                // Keep top_k by score
                neigh.sort_by(|x, y| y.1.partial_cmp(&x.1).unwrap_or(std::cmp::Ordering::Equal));
                neigh.truncate(top_k);

                // Add edges (bidirectional)
                for (j, score) in neigh {
                    self.graph.edges.push(GraphEdgeEntry {
                        from: self.chunks[i].chunk_id.clone(),
                        to: self.chunks[j].chunk_id.clone(),
                        edge_type: "semantic_chunk_similarity".to_string(),
                        score: Some(score),
                    });
                }
            }

            println!("⚡ Computing symbol embeddings...");
            // 2) Compute symbol embeddings as mean of their chunk embeddings
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
                        // Use hash map lookup instead of linear search
                        if let Some(&pos) = chunk_id_to_idx.get(cid) {
                            if pos < self.embeddings.len() {
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
                    }
                    if count > 0 {
                        for v in agg.iter_mut() {
                            *v /= count as f32;
                        }
                        // normalize symbol embedding
                        Self::normalize_inplace(&mut agg);
                        symbol_embeddings
                            .insert(format!("symbol:{}:{}", pf.file_node.path, symbol.name), agg);
                        symbol_chunk_counts.insert(
                            format!("symbol:{}:{}", pf.file_node.path, symbol.name),
                            count,
                        );
                    }
                }
            }

            println!("⚡ Computing symbol-to-symbol similarities...");
            // 3) Symbol → Symbol similarity (top-3) - skip if too many symbols
            let symbol_ids: Vec<String> = symbol_embeddings.keys().cloned().collect();
            let n_symbols = symbol_ids.len();

            if n_symbols <= 500 {
                // Only compute if reasonable number
                for (idx, sid) in symbol_ids.iter().enumerate() {
                    let a = &symbol_embeddings[sid];
                    let mut neigh: Vec<(String, f32)> = Vec::new();
                    for (_jdx, sj) in symbol_ids.iter().enumerate().skip(idx + 1) {
                        let b = &symbol_embeddings[sj];
                        let score = Self::cosine(a.as_slice(), b.as_slice());
                        neigh.push((sj.clone(), score));
                    }
                    neigh
                        .sort_by(|x, y| y.1.partial_cmp(&x.1).unwrap_or(std::cmp::Ordering::Equal));
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
            } else {
                println!(
                    "⚠️  Skipping symbol-to-symbol similarity (too many symbols: {})",
                    n_symbols
                );
            }

            println!("⚡ Computing symbol-to-chunk links...");
            // 4) Symbol → Chunk top-3 links (by cosine) - skip if product is too large
            if n_symbols * n_chunks <= 50000 {
                // Only compute if reasonable
                for (sid, emb) in &symbol_embeddings {
                    let mut neigh: Vec<(String, f32)> = Vec::new();
                    for i in 0..n_chunks {
                        let score = Self::cosine(emb.as_slice(), self.embeddings[i].as_ref());
                        neigh.push((self.chunks[i].chunk_id.clone(), score));
                    }
                    neigh
                        .sort_by(|x, y| y.1.partial_cmp(&x.1).unwrap_or(std::cmp::Ordering::Equal));
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
            } else {
                println!(
                    "⚠️  Skipping symbol-to-chunk links (too large: {} symbols × {} chunks)",
                    n_symbols, n_chunks
                );
            }

            // 5) Clustering (DBSCAN) on chunk embeddings
            // Use cosine distance: distance = 1 - cosine_similarity (embeddings are normalized)
            let eps = 0.3_f32; // distance threshold (1 - cosine). corresponds to cosine ~0.7
            let min_points = 3usize;
            let assignments = Self::dbscan_cluster(&self.embeddings, eps, min_points);

            // annotate chunks with cluster metadata
            for (i, cluster_opt) in assignments.iter().enumerate() {
                if let Some(node) = self
                    .graph
                    .nodes
                    .iter_mut()
                    .find(|n| n.id == self.chunks[i].chunk_id)
                {
                    match cluster_opt {
                        Some(cid) => {
                            node.metadata.insert("cluster".to_string(), cid.to_string());
                        }
                        None => {
                            node.metadata
                                .insert("cluster".to_string(), "noise".to_string());
                        }
                    }
                }
            }

            println!("⚡ Propagating cluster assignments to symbols...");
            // propagate cluster membership to symbols (majority of their chunks, ignoring noise)
            for pf in processed_files {
                for symbol in &pf.symbols {
                    if symbol.chunk_ids.is_empty() {
                        continue;
                    }
                    let mut counts: HashMap<usize, usize> = HashMap::new();
                    let mut total = 0usize;
                    for cid in &symbol.chunk_ids {
                        // Use hash map lookup instead of linear search
                        if let Some(&pos) = chunk_id_to_idx.get(cid) {
                            if pos < assignments.len() {
                                if let Some(cl) = assignments[pos] {
                                    *counts.entry(cl).or_default() += 1;
                                    total += 1;
                                }
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
                        node.metadata
                            .insert("cluster".to_string(), best_cluster.to_string());
                        node.metadata
                            .insert("cluster_confidence".to_string(), format!("{:.3}", conf));
                    }
                }
            }
        }
    }

    fn generate_documentation(&mut self, processed_files: &[ProcessedFile]) {
        // Use comprehensive docgen module for documentation generation
        println!("🔍 Generating comprehensive documentation from graph...");

        let docgen = DocGenerator::new(processed_files);
        let generated_docs = docgen.generate_all();

        // Convert generated docs to serializable format
        self.documentation.module_summaries = generated_docs
            .module_summaries
            .iter()
            .map(|m| ModuleSummaryData {
                module_path: m.module_path.clone(),
                description: m.description.clone(),
                file_count: m.file_count,
                symbol_count: m.symbol_count,
                lines_of_code: m.lines_of_code,
                primary_purpose: m.primary_purpose.clone(),
            })
            .collect();

        self.documentation.file_summaries = generated_docs
            .file_summaries
            .iter()
            .map(|f| FileSummaryData {
                file_path: f.file_path.clone(),
                description: f.description.clone(),
                language: f.language.clone(),
                lines_total: f.lines_total,
                lines_code: f.lines_code,
                complexity_score: f.complexity_score,
            })
            .collect();

        self.documentation.struct_docs = generated_docs
            .struct_docs
            .iter()
            .map(|s| StructDocData {
                name: s.name.clone(),
                file_path: s.file_path.clone(),
                description: s.description.clone(),
                visibility: s.visibility.clone(),
                field_count: s.fields.len(),
                method_count: s.methods.len(),
            })
            .collect();

        self.documentation.function_docs = generated_docs
            .function_docs
            .iter()
            .map(|f| FunctionDocData {
                name: f.name.clone(),
                file_path: f.file_path.clone(),
                description: f.description.clone(),
                signature: f.signature.clone(),
                complexity_estimate: f.complexity_estimate.clone(),
            })
            .collect();

        self.documentation.dependency_overview = DependencyOverviewData {
            internal_count: generated_docs
                .dependency_overview
                .internal_dependencies
                .len(),
            external_count: generated_docs
                .dependency_overview
                .external_dependencies
                .len(),
            circular_count: generated_docs
                .dependency_overview
                .circular_dependencies
                .len(),
            summary: generated_docs
                .dependency_overview
                .dependency_graph_summary
                .clone(),
        };

        self.documentation.cluster_summaries = generated_docs
            .cluster_summaries
            .iter()
            .map(|c| ClusterSummaryData {
                cluster_id: c.cluster_id.clone(),
                topic_label: c.topic_label.clone(),
                description: c.description.clone(),
                symbol_count: c.symbol_count,
            })
            .collect();

        // Generate architecture overview from comprehensive data
        let arch = &generated_docs.architecture_overview;
        self.documentation.architecture_overview = format!(
            "Architecture Style: {}\n\nTotal Files: {}\nTotal Symbols: {}\nTotal Lines: {}\n\nLanguages: {:?}\n\nCore Components: {}\n\nDesign Patterns: {}\n\nModule Hierarchy: {}",
            arch.architectural_style,
            arch.total_files,
            arch.total_symbols,
            arch.total_lines,
            arch.language_breakdown,
            arch.core_components.len(),
            arch.design_patterns.join(", "),
            arch.module_hierarchy
        );

        // Generate highlights
        self.documentation.highlights.push(format!(
            "📁 {} modules analyzed",
            self.documentation.module_summaries.len()
        ));
        self.documentation.highlights.push(format!(
            "📄 {} files documented",
            self.documentation.file_summaries.len()
        ));
        self.documentation.highlights.push(format!(
            "🏗️  {} structs documented",
            self.documentation.struct_docs.len()
        ));
        self.documentation.highlights.push(format!(
            "⚙️  {} functions documented",
            self.documentation.function_docs.len()
        ));
        self.documentation.highlights.push(format!(
            "🔗 {} internal dependencies",
            self.documentation.dependency_overview.internal_count
        ));
        self.documentation.highlights.push(format!(
            "📊 {} semantic clusters identified",
            self.documentation.cluster_summaries.len()
        ));

        // Keep legacy symbol summaries for compatibility
        for pf in processed_files {
            for symbol in &pf.symbols {
                let summary = self.generate_symbol_summary(symbol);

                self.documentation.summaries.push(SymbolSummary {
                    symbol_id: format!("symbol:{}:{}", pf.file_node.path, symbol.name),
                    summary,
                    details: symbol.docs.clone().unwrap_or_default(),
                    related: Vec::new(),
                });
            }
        }

        println!("✅ Generated comprehensive documentation:");
        println!(
            "   - {} module summaries",
            self.documentation.module_summaries.len()
        );
        println!(
            "   - {} file summaries",
            self.documentation.file_summaries.len()
        );
        println!("   - {} struct docs", self.documentation.struct_docs.len());
        println!(
            "   - {} function docs",
            self.documentation.function_docs.len()
        );
        println!(
            "   - {} cluster summaries",
            self.documentation.cluster_summaries.len()
        );
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
    pub fn write_to_file<P: AsRef<Path>>(
        &self,
        output_path: P,
        project_graph: Option<&crate::ingest::ProjectGraph>,
    ) -> Result<()> {
        self.write_to_file_with_llm(output_path, project_graph, None)
    }

    /// Write the .docpack file to disk with optional LLM summary generation
    pub fn write_to_file_with_llm<P: AsRef<Path>>(
        &self,
        output_path: P,
        project_graph: Option<&crate::ingest::ProjectGraph>,
        llm_engine: Option<&mut crate::llm::LlmEngine>,
    ) -> Result<()> {
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

        // Write agent_index.json (if project graph is available)
        if let Some(pg) = project_graph {
            zip.start_file("agent_index.json", options)?;
            let agent_index = crate::agent::build_agent_index_with_llm(pg, llm_engine);
            let agent_index_json = serde_json::to_string_pretty(&agent_index)?;
            zip.write_all(agent_index_json.as_bytes())?;
        }

        // Write embeddings.bin (binary format: row-major f32 array)
        zip.start_file("embeddings.bin", options)?;
        for embedding in &self.embeddings {
            for &value in embedding.as_ref() {
                zip.write_all(&value.to_le_bytes())?;
            }
        }

        // Write source files to source/ directory
        for pf in &self.processed_files {
            let source_path = format!("source/{}", pf.file_node.path.trim_start_matches('/'));
            zip.start_file(source_path, options)?;
            zip.write_all(&pf.original_bytes)?;
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
