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
    embeddings: Vec<Vec<f32>>,
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
            for &value in embedding {
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
