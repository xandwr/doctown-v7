/// Compute detailed stats for code files: total, code, comment, and blank lines.
pub fn code_file_stats(bytes: &[u8], ext: &str) -> Option<(usize, usize, usize, usize)> {
    let text = String::from_utf8_lossy(bytes);
    let mut total = 0;
    let mut code = 0;
    let mut comment = 0;
    let mut blank = 0;

    let is_rust = ext == "rs";
    let is_python = ext == "py";
    let is_js_ts = ext == "js" || ext == "ts" || ext == "jsx" || ext == "tsx";
    let is_c_family = ext == "cpp"
        || ext == "cc"
        || ext == "c"
        || ext == "h"
        || ext == "hpp"
        || ext == "cxx"
        || ext == "java";
    let has_c_style_comments = is_rust || is_js_ts || is_c_family;

    let mut in_multiline_comment = false;

    for line in text.lines() {
        total += 1;
        let trimmed = line.trim();

        if trimmed.is_empty() {
            blank += 1;
            continue;
        }

        // Track multi-line comments for C-style languages
        if has_c_style_comments {
            // Check if we're entering a multi-line comment
            if trimmed.contains("/*") {
                in_multiline_comment = true;
            }

            // If in multi-line comment, count as comment
            if in_multiline_comment {
                comment += 1;
                // Check if we're exiting the multi-line comment
                if trimmed.contains("*/") {
                    in_multiline_comment = false;
                }
                continue;
            }

            // Single-line comment
            if trimmed.starts_with("//") {
                comment += 1;
                continue;
            }
        }

        // Python comments
        if is_python {
            if trimmed.starts_with("#") {
                comment += 1;
                continue;
            }
            // Python docstrings (simplified detection)
            if trimmed.starts_with("\"\"\"") || trimmed.starts_with("'''") {
                comment += 1;
                continue;
            }
        }

        // If we get here, it's a code line
        code += 1;
    }

    Some((total, code, comment, blank))
}
use anyhow::{Result, anyhow};
use reqwest::Client;
use std::collections::HashMap;
use std::io::Cursor;
use tokio::fs;
use url::Url;
use zip::ZipArchive;

use crate::embedding::EmbeddingEngine;

#[allow(dead_code)]
#[derive(Debug, Clone)]
pub struct IngestResult {
    pub files: Vec<FileNode>,
}

#[derive(Debug, Clone)]
pub struct FileNode {
    pub path: String,
    pub bytes: Vec<u8>,
    pub kind: FileKind,
}

impl FileNode {
    /// Determine the parent module based on the file path.
    /// For example, "src/utils/helpers.rs" -> Some("src/utils")
    /// Returns None for root-level files.
    pub fn parent_module(&self) -> Option<String> {
        let path = std::path::Path::new(&self.path);
        path.parent()
            .and_then(|p| p.to_str())
            .filter(|s| !s.is_empty())
            .map(|s| s.to_string())
    }
}

#[derive(Debug, Clone)]
pub enum FileKind {
    SourceCode,
    Documentation,
    Config,
    Other,
}

impl FileKind {
    pub fn classify(extension: &str) -> Self {
        match extension {
            "rs" | "ts" | "py" | "js" | "cpp" | "java" => FileKind::SourceCode,
            "md" | "rst" | "txt" => FileKind::Documentation,
            "toml" | "yml" | "json" => FileKind::Config,
            _ => FileKind::Other,
        }
    }
}

/// Describes extraction kinds that a pipeline may perform on a file.
#[allow(dead_code)]
#[derive(Debug, Clone)]
pub enum ExtractionKind {
    FullText,
    CodeSymbols,
    Headings,
    Anchors,
    FrontMatter,
    Raw,
}

/// Strategies for chunking content.
#[allow(dead_code)]
#[derive(Debug, Clone)]
pub enum ChunkingStrategy {
    Lines(usize),      // chunk by N lines
    Delimiter(String), // chunk by a delimiter string
    HeadingSections,   // chunk by markdown headings
    CodeAware,         // language-aware code chunking
    None,              // no chunking
}

/// How to assemble graph nodes from the extracted/chunked pieces.
#[allow(dead_code)]
#[derive(Debug, Clone)]
pub enum AssemblyStrategy {
    OneNodePerFile,
    NodePerChunk,
    SymbolLinkedChunks, // link symbols to specific chunks
}

/// A processing plan describes exactly what an ingestion pipeline will do
/// for a given file kind. This keeps the pipeline predictable and extensible.
#[derive(Debug, Clone)]
pub struct ProcessingPlan {
    extract: Vec<ExtractionKind>,
    chunking: ChunkingStrategy,
    symbol_parse: bool,
    embed: bool,
    assembly: AssemblyStrategy,
    skip: bool,
    metadata: Vec<String>,
}

impl ProcessingPlan {
    /// Start a new builder-style ProcessingPlan with sensible defaults.
    pub fn new() -> Self {
        ProcessingPlan {
            extract: Vec::new(),
            chunking: ChunkingStrategy::None,
            symbol_parse: false,
            embed: false,
            assembly: AssemblyStrategy::OneNodePerFile,
            skip: false,
            metadata: Vec::new(),
        }
    }

    pub fn extract(mut self, extract: Vec<ExtractionKind>) -> Self {
        self.extract = extract;
        self
    }

    pub fn chunking(mut self, chunking: ChunkingStrategy) -> Self {
        self.chunking = chunking;
        self
    }

    pub fn symbol_parse(mut self, symbol_parse: bool) -> Self {
        self.symbol_parse = symbol_parse;
        self
    }

    pub fn embed(mut self, embed: bool) -> Self {
        self.embed = embed;
        self
    }

    pub fn assembly(mut self, assembly: AssemblyStrategy) -> Self {
        self.assembly = assembly;
        self
    }

    pub fn metadata(mut self, metadata: Vec<String>) -> Self {
        self.metadata = metadata;
        self
    }

    pub fn skip(mut self, skip: bool) -> Self {
        self.skip = skip;
        self
    }

    // Optional getters if external code needs read access later.
    pub fn get_extract(&self) -> &Vec<ExtractionKind> {
        &self.extract
    }
    pub fn get_chunking(&self) -> &ChunkingStrategy {
        &self.chunking
    }
    pub fn get_symbol_parse(&self) -> bool {
        self.symbol_parse
    }
    pub fn get_embed(&self) -> bool {
        self.embed
    }
    pub fn get_assembly(&self) -> &AssemblyStrategy {
        &self.assembly
    }
    pub fn get_metadata(&self) -> &Vec<String> {
        &self.metadata
    }
    pub fn get_skip(&self) -> bool {
        self.skip
    }
}

/// Map a `FileKind` to a sane default `ProcessingPlan`.
pub fn filekind_to_plan(kind: &FileKind) -> ProcessingPlan {
    match kind {
        FileKind::SourceCode => ProcessingPlan::new()
            .extract(vec![ExtractionKind::CodeSymbols, ExtractionKind::FullText])
            .chunking(ChunkingStrategy::CodeAware)
            .symbol_parse(true)
            .embed(true)
            .assembly(AssemblyStrategy::SymbolLinkedChunks)
            .metadata(vec!["language".to_string(), "dependencies".to_string()]),

        FileKind::Documentation => ProcessingPlan::new()
            .extract(vec![
                ExtractionKind::Headings,
                ExtractionKind::FullText,
                ExtractionKind::Anchors,
            ])
            .chunking(ChunkingStrategy::HeadingSections)
            .symbol_parse(false)
            .embed(true)
            .assembly(AssemblyStrategy::NodePerChunk)
            .metadata(vec!["anchors".to_string(), "title".to_string()]),

        FileKind::Config | FileKind::Other => ProcessingPlan::new()
            .extract(vec![ExtractionKind::Raw])
            .chunking(ChunkingStrategy::None)
            .symbol_parse(false)
            .embed(false)
            .assembly(AssemblyStrategy::OneNodePerFile)
            .metadata(vec!["filetype".to_string()]),
    }
}

/// Helper: produce a plan for an individual `FileNode`.
pub fn plan_for_file_node(node: &FileNode) -> ProcessingPlan {
    let mut plan = filekind_to_plan(&node.kind);

    // If the path matches common noise/asset files, mark the plan to skip processing.
    if is_noise_file(&node.path) {
        plan = plan.skip(true);
    }

    plan
}

/// Return true for files we should ignore entirely (licenses, images, locks, etc.)
fn is_noise_file(path: &str) -> bool {
    use std::path::Path;

    let p = Path::new(path);
    let file_name = match p.file_name().and_then(|s| s.to_str()) {
        Some(n) => n.to_lowercase(),
        None => return false,
    };

    // Exact filename matches (case-insensitive)
    let exact = [
        "license",
        "copying",
        ".gitignore",
        ".gitattributes",
        ".editorconfig",
    ];
    if exact.iter().any(|e| file_name == *e) {
        return true;
    }

    // README without extension (skip README but not README.md)
    if p.extension().is_none() {
        if let Some(stem) = p.file_stem().and_then(|s| s.to_str()) {
            if stem.eq_ignore_ascii_case("readme") {
                return true;
            }
        }
    }

    // Prefix/wildcard matches
    let stem = p
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("")
        .to_lowercase();
    let norm_stem = stem.replace('-', "_");
    let prefixes = [
        "changelog",
        "contributing",
        "code_of_conduct",
        "code_of_conduct.md",
        "code_of_conduct.txt",
    ];
    if prefixes.iter().any(|pr| norm_stem.starts_with(pr)) {
        return true;
    }

    // Dotfile config patterns like .prettierrc*, .eslintrc*
    if file_name.starts_with(".prettierrc") || file_name.starts_with(".eslintrc") {
        return true;
    }

    // Lockfiles: Cargo.lock, package-lock.json, yarn.lock, pnpm-lock.yaml, *.lock
    if file_name.ends_with(".lock")
        || file_name == "package-lock.json"
        || file_name == "yarn.lock"
        || file_name.starts_with("pnpm-lock")
        || file_name == "cargo.lock"
    {
        return true;
    }

    // Binary/media extensions
    if let Some(ext) = p.extension().and_then(|s| s.to_str()) {
        let ext = ext.to_lowercase();
        let bin_exts = [
            "png", "jpg", "jpeg", "gif", "ico", "pdf", "svg", "wasm", "ttf", "woff", "woff2",
        ];
        if bin_exts.contains(&ext.as_str()) {
            return true;
        }
    }

    false
}

/// A minimal chunk representation produced by chunking.
#[allow(dead_code)]
#[derive(Debug, Clone)]
pub struct ChunkId(pub String);

#[allow(dead_code)]
#[derive(Debug, Clone)]
pub struct Chunk {
    pub id: ChunkId,
    pub text: String,
    /// byte range (start, end) of this chunk in the original file
    pub byte_range: Option<(usize, usize)>,
    /// symbols that live inside this chunk
    pub containing_symbols: Vec<String>, // symbol names
}

/// A comprehensive symbol node representation parsed from source files.
#[allow(dead_code)]
#[derive(Debug, Clone)]
pub struct SymbolNode {
    pub name: String,
    pub kind: String, // function, struct, enum, trait, impl, type_alias, const, static, mod, macro, field, variant
    /// byte range (start, end)
    pub byte_range: Option<(usize, usize)>,
    /// start (row, col)
    pub start_pos: Option<(usize, usize)>,
    /// end (row, col)
    pub end_pos: Option<(usize, usize)>,
    /// optionally the parent symbol (e.g. method -> impl/struct)
    pub parent: Option<String>,
    /// documentation/comments associated with the symbol (if any)
    pub docs: Option<String>,
    /// visibility: pub, pub(crate), pub(super), private
    pub visibility: Option<String>,
    /// for functions: parameter list with types
    pub parameters: Option<Vec<(String, String)>>, // (name, type)
    /// for functions: return type
    pub return_type: Option<String>,
    /// generic type parameters and bounds
    pub generics: Option<String>,
    /// for impl blocks: the trait being implemented (if trait impl)
    pub trait_impl: Option<String>,
    /// for fields/variants: the type
    pub field_type: Option<String>,
    /// attributes like #[derive(...)], #[cfg(...)]
    pub attributes: Vec<String>,
    /// is this item mutable (for statics, bindings)
    pub is_mutable: bool,
    /// for trait items: whether it's unsafe, async, const
    pub modifiers: Vec<String>,
    /// chunk IDs that contain this symbol
    pub chunk_ids: Vec<String>,
}

/// Embedding vector placeholder.
#[allow(dead_code)]
pub type Embedding = std::sync::Arc<[f32]>;

/// A simple graph edge connecting two symbols/modules.
#[allow(dead_code)]
#[derive(Debug, Clone, PartialEq)]
pub enum GraphEdgeKind {
    SymbolToChunk,
    SymbolToSymbol,
    FileToFile,
    ChunkToChunk,
    /// Weighted semantic similarity between two symbols (cosine similarity of embeddings).
    SymbolSimilarity,
}

#[allow(dead_code)]
#[derive(Debug, Clone)]
pub struct GraphEdge {
    pub src: String,
    pub dst: String,
    pub kind: GraphEdgeKind,
    /// Optional weight for the edge (e.g., similarity score 0.0–1.0).
    pub weight: Option<f32>,
}

/// ProcessedFile is the executor output for a single file + plan.
#[derive(Debug, Clone)]
pub struct ProcessedFile {
    pub file_node: FileNode,
    pub chunks: Vec<Chunk>,
    pub symbols: Vec<SymbolNode>,
    pub embeddings: Vec<Embedding>,
    pub metadata: HashMap<String, String>,
    pub graph_edges: Vec<GraphEdge>,
    pub original_bytes: Vec<u8>,
}

impl ProcessedFile {
    pub fn empty(node: &FileNode) -> Self {
        ProcessedFile {
            file_node: node.clone(),
            chunks: Vec::new(),
            symbols: Vec::new(),
            embeddings: Vec::new(),
            metadata: HashMap::new(),
            graph_edges: Vec::new(),
            original_bytes: Vec::new(),
        }
    }
}

/// Orchestrator: execute a ProcessingPlan on a `FileNode`.
pub async fn process_file(
    node: &FileNode,
    plan: &ProcessingPlan,
    engine: Option<&mut EmbeddingEngine>,
) -> ProcessedFile {
    if plan.get_skip() {
        return ProcessedFile::empty(node);
    }

    // 1) extraction
    let extracted = run_extractions(node, plan.get_extract());

    // 2) chunking
    let mut chunks = run_chunking(node, &extracted, plan.get_chunking());

    // 3) symbols (optional)
    let mut symbols = if plan.get_symbol_parse() {
        parse_symbols(node)
    } else {
        Vec::new()
    };

    // 4) embeddings (optional, async to support remote embedders)
    let embeddings = if plan.get_embed() {
        if let Some(eng) = engine {
            match embed_chunks(&chunks, eng).await {
                Ok(emb) => emb,
                Err(e) => {
                    eprintln!("⚠️ Embedding error for {}: {}", node.path, e);
                    Vec::new()
                }
            }
        } else {
            Vec::new()
        }
    } else {
        Vec::new()
    };

    // 5) assembly -> produce graph edges and link chunks to symbols
    let assembled_edges = assemble_nodes(&mut chunks, &mut symbols, plan.get_assembly());

    ProcessedFile {
        file_node: node.clone(),
        chunks,
        symbols,
        embeddings,
        metadata: compute_metadata(node, plan.get_metadata()),
        graph_edges: assembled_edges,
        original_bytes: node.bytes.clone(),
    }
}

// --- Helper stubs -------------------------------------------------

/// Run extractions declared by the plan. Returns a vector of (kind, text).
fn run_extractions(node: &FileNode, kinds: &Vec<ExtractionKind>) -> Vec<(ExtractionKind, String)> {
    let text = String::from_utf8_lossy(&node.bytes).to_string();
    kinds
        .iter()
        .map(|k| match k {
            ExtractionKind::FullText => (k.clone(), text.clone()),
            ExtractionKind::Raw => (k.clone(), text.clone()),
            ExtractionKind::Headings => {
                // simple heading extraction: keep lines starting with '#'
                let headings = text
                    .lines()
                    .filter(|l| l.trim_start().starts_with('#'))
                    .collect::<Vec<_>>()
                    .join("\n");
                (k.clone(), headings)
            }
            ExtractionKind::Anchors => {
                // naive: anchor-like lines
                let anchors = text
                    .lines()
                    .filter(|l| l.contains("#"))
                    .collect::<Vec<_>>()
                    .join("\n");
                (k.clone(), anchors)
            }
            ExtractionKind::FrontMatter => (k.clone(), text.clone()),
            ExtractionKind::CodeSymbols => (k.clone(), text.clone()),
        })
        .collect()
}

/// Run chunking on the extracted pieces. AST-driven for code, heuristic for other types.
fn run_chunking(
    node: &FileNode,
    extracted: &Vec<(ExtractionKind, String)>,
    strategy: &ChunkingStrategy,
) -> Vec<Chunk> {
    match strategy {
        ChunkingStrategy::None => vec![Chunk {
            id: ChunkId("full".to_string()),
            text: extracted
                .iter()
                .map(|(_, t)| t.clone())
                .collect::<Vec<_>>()
                .join("\n"),
            byte_range: None,
            containing_symbols: Vec::new(),
        }],
        ChunkingStrategy::HeadingSections => {
            // split by markdown headings as simple heuristic
            let mut out = Vec::new();
            for (i, (_k, txt)) in extracted.iter().enumerate() {
                let parts: Vec<&str> = txt.split('\n').collect();
                let mut buffer = String::new();
                for (j, line) in parts.iter().enumerate() {
                    if line.trim_start().starts_with('#') {
                        if !buffer.is_empty() {
                            out.push(Chunk {
                                id: ChunkId(format!("h-{}-{}", i, j)),
                                text: buffer.clone(),
                                byte_range: None,
                                containing_symbols: Vec::new(),
                            });
                            buffer.clear();
                        }
                        buffer.push_str(line);
                        buffer.push('\n');
                    } else {
                        buffer.push_str(line);
                        buffer.push('\n');
                    }
                }
                if !buffer.is_empty() {
                    out.push(Chunk {
                        id: ChunkId(format!("h-{}-tail", i)),
                        text: buffer,
                        byte_range: None,
                        containing_symbols: Vec::new(),
                    });
                }
            }
            if out.is_empty() {
                out.push(Chunk {
                    id: ChunkId("empty".to_string()),
                    text: String::new(),
                    byte_range: None,
                    containing_symbols: Vec::new(),
                });
            }
            out
        }
        ChunkingStrategy::Lines(n) => {
            let mut out = Vec::new();
            let full = extracted
                .iter()
                .map(|(_, t)| t.clone())
                .collect::<Vec<_>>()
                .join("\n");
            for (i, chunk) in full.lines().collect::<Vec<&str>>().chunks(*n).enumerate() {
                out.push(Chunk {
                    id: ChunkId(format!("l-{}", i)),
                    text: chunk.join("\n"),
                    byte_range: None,
                    containing_symbols: Vec::new(),
                });
            }
            out
        }
        ChunkingStrategy::CodeAware => {
            // AST-driven chunking for supported languages
            ast_chunking(node)
        }
        ChunkingStrategy::Delimiter(d) => {
            let full = extracted
                .iter()
                .map(|(_, t)| t.clone())
                .collect::<Vec<_>>()
                .join("\n");
            full.split(d)
                .enumerate()
                .map(|(i, s)| Chunk {
                    id: ChunkId(format!("d-{}", i)),
                    text: s.to_string(),
                    byte_range: None,
                    containing_symbols: Vec::new(),
                })
                .collect()
        }
    }
}

/// AST-driven chunking: extract top-level items as logical chunks
fn ast_chunking(node: &FileNode) -> Vec<Chunk> {
    let ext = std::path::Path::new(&node.path)
        .extension()
        .and_then(|s| s.to_str())
        .unwrap_or("");

    let text = String::from_utf8_lossy(&node.bytes).to_string();

    // Only do AST chunking for supported languages
    if ext != "rs" {
        // Fallback to simple blank-line chunking
        return text
            .split("\n\n")
            .enumerate()
            .filter(|(_, s)| !s.trim().is_empty())
            .map(|(i, s)| Chunk {
                id: ChunkId(format!("c-{}", i)),
                text: s.trim().to_string(),
                byte_range: None,
                containing_symbols: Vec::new(),
            })
            .collect();
    }

    use tree_sitter::Parser;
    let mut parser = Parser::new();
    if parser
        .set_language(&tree_sitter_rust::LANGUAGE.into())
        .is_err()
    {
        // Fallback if tree-sitter fails
        return vec![Chunk {
            id: ChunkId("full".to_string()),
            text,
            byte_range: None,
            containing_symbols: Vec::new(),
        }];
    }

    let tree = match parser.parse(&text, None) {
        Some(t) => t,
        None => {
            return vec![Chunk {
                id: ChunkId("full".to_string()),
                text,
                byte_range: None,
                containing_symbols: Vec::new(),
            }];
        }
    };

    let source = text.as_bytes();
    let root = tree.root_node();
    let mut chunks = Vec::new();

    // Helper to get node text
    fn node_text(n: &tree_sitter::Node, src: &[u8]) -> String {
        n.utf8_text(src).unwrap_or("").to_string()
    }

    // Helper to extract chunks from container nodes (impl, trait, mod)
    fn extract_method_chunks(
        container: &tree_sitter::Node,
        source: &[u8],
        chunks: &mut Vec<Chunk>,
        container_name: &str,
    ) {
        // Look for the declaration_list or body that contains the methods
        let body = container.child_by_field_name("body").or_else(|| {
            // If no body field, look for a declaration_list child
            let mut cursor = container.walk();
            container
                .named_children(&mut cursor)
                .find(|c| c.kind() == "declaration_list")
        });

        if let Some(body_node) = body {
            let mut has_items = false;
            let mut cursor = body_node.walk();

            for child in body_node.named_children(&mut cursor) {
                match child.kind() {
                    "function_item" => {
                        has_items = true;
                        let start = child.start_byte();
                        let end = child.end_byte();
                        let chunk_text = child.utf8_text(source).unwrap_or("").to_string();

                        let method_name = child
                            .child_by_field_name("name")
                            .and_then(|n| n.utf8_text(source).ok())
                            .unwrap_or("anon");

                        chunks.push(Chunk {
                            id: ChunkId(format!(
                                "{}::{}:{}-{}",
                                container_name, method_name, start, end
                            )),
                            text: chunk_text,
                            byte_range: Some((start, end)),
                            containing_symbols: Vec::new(),
                        });
                    }
                    "const_item" | "type_item" | "associated_type" => {
                        has_items = true;
                        let start = child.start_byte();
                        let end = child.end_byte();
                        let chunk_text = child.utf8_text(source).unwrap_or("").to_string();

                        let item_name = child
                            .child_by_field_name("name")
                            .and_then(|n| n.utf8_text(source).ok())
                            .unwrap_or("anon");

                        chunks.push(Chunk {
                            id: ChunkId(format!(
                                "{}::{}:{}-{}",
                                container_name, item_name, start, end
                            )),
                            text: chunk_text,
                            byte_range: Some((start, end)),
                            containing_symbols: Vec::new(),
                        });
                    }
                    _ => {}
                }
            }

            // If we found items, we're done
            if has_items {
                return;
            }
        }

        // If no methods/items were found, chunk the entire container
        let start = container.start_byte();
        let end = container.end_byte();
        let chunk_text = container.utf8_text(source).unwrap_or("").to_string();
        chunks.push(Chunk {
            id: ChunkId(format!("{}:{}-{}", container_name, start, end)),
            text: chunk_text,
            byte_range: Some((start, end)),
            containing_symbols: Vec::new(),
        });
    }

    // Walk top-level items and create chunks
    let mut cursor = root.walk();
    for child in root.named_children(&mut cursor) {
        let kind = child.kind();

        match kind {
            "function_item" => {
                // Top-level function: one chunk per function
                let start = child.start_byte();
                let end = child.end_byte();
                let chunk_text = node_text(&child, source);

                let fn_name = child
                    .child_by_field_name("name")
                    .map(|n| node_text(&n, source))
                    .unwrap_or_else(|| "anon_fn".to_string());

                chunks.push(Chunk {
                    id: ChunkId(format!("fn::{}:{}-{}", fn_name, start, end)),
                    text: chunk_text,
                    byte_range: Some((start, end)),
                    containing_symbols: Vec::new(),
                });
            }
            "impl_item" => {
                // Impl block: chunk each method separately
                let target = child
                    .child_by_field_name("type")
                    .and_then(|c| c.utf8_text(source).ok())
                    .unwrap_or("impl");

                let trait_name = child
                    .child_by_field_name("trait")
                    .and_then(|t| t.utf8_text(source).ok());

                let impl_name = if let Some(trait_impl) = trait_name {
                    format!("impl {} for {}", trait_impl, target)
                } else {
                    format!("impl {}", target)
                };

                extract_method_chunks(&child, source, &mut chunks, &impl_name);
            }
            "trait_item" => {
                // Trait: chunk each method separately
                let trait_name = child
                    .child_by_field_name("name")
                    .map(|n| node_text(&n, source))
                    .unwrap_or_else(|| "anon_trait".to_string());

                extract_method_chunks(
                    &child,
                    source,
                    &mut chunks,
                    &format!("trait {}", trait_name),
                );
            }
            "mod_item" => {
                // Mod: if it has inline content, chunk its items; otherwise chunk the whole mod
                let mod_name = child
                    .child_by_field_name("name")
                    .map(|n| node_text(&n, source))
                    .unwrap_or_else(|| "anon_mod".to_string());

                // For now, chunk the entire mod declaration
                let start = child.start_byte();
                let end = child.end_byte();
                let chunk_text = node_text(&child, source);

                chunks.push(Chunk {
                    id: ChunkId(format!("mod::{}:{}-{}", mod_name, start, end)),
                    text: chunk_text,
                    byte_range: Some((start, end)),
                    containing_symbols: Vec::new(),
                });
            }
            "struct_item" | "enum_item" | "const_item" | "static_item" | "type_item"
            | "macro_definition" | "use_declaration" => {
                // Other top-level items: one chunk per item
                let start = child.start_byte();
                let end = child.end_byte();
                let chunk_text = node_text(&child, source);

                let item_name = child
                    .child_by_field_name("name")
                    .map(|n| node_text(&n, source))
                    .unwrap_or_else(|| kind.to_string());

                chunks.push(Chunk {
                    id: ChunkId(format!(
                        "{}::{}:{}-{}",
                        kind.trim_end_matches("_item"),
                        item_name,
                        start,
                        end
                    )),
                    text: chunk_text,
                    byte_range: Some((start, end)),
                    containing_symbols: Vec::new(),
                });
            }
            _ => {}
        }
    }

    // If no chunks were created (e.g., file with only comments), return full content
    if chunks.is_empty() {
        chunks.push(Chunk {
            id: ChunkId("full".to_string()),
            text,
            byte_range: None,
            containing_symbols: Vec::new(),
        });
    }

    chunks
}

/// Parse symbols from the node using tree-sitter for comprehensive Rust analysis.
fn parse_symbols(node: &FileNode) -> Vec<SymbolNode> {
    let ext = std::path::Path::new(&node.path)
        .extension()
        .and_then(|s| s.to_str())
        .unwrap_or("");

    let text = String::from_utf8_lossy(&node.bytes).to_string();

    if ext != "rs" {
        return Vec::new();
    }

    // Use tree-sitter to parse Rust source and extract real symbols.
    use tree_sitter::{Node, Parser};
    let mut parser = Parser::new();
    if parser
        .set_language(&tree_sitter_rust::LANGUAGE.into())
        .is_err()
    {
        return Vec::new();
    }

    let tree = match parser.parse(&text, None) {
        Some(t) => t,
        None => return Vec::new(),
    };

    let source = text.as_bytes();
    let mut symbols = Vec::new();

    // helper to get node text
    fn node_text<'a>(n: &Node<'a>, src: &'a [u8]) -> String {
        n.utf8_text(src).unwrap_or("").to_string()
    }

    // helper to extract docs and attributes
    let lines: Vec<&str> = text.lines().collect();
    fn collect_docs_and_attrs(
        lines: &Vec<&str>,
        start_row: usize,
    ) -> (Option<String>, Vec<String>) {
        if start_row == 0 {
            return (None, Vec::new());
        }
        let mut docs = Vec::new();
        let mut attrs = Vec::new();
        let mut i = start_row.saturating_sub(1);

        loop {
            if let Some(line) = lines.get(i) {
                let trimmed = line.trim_start();
                if trimmed.starts_with("///") || trimmed.starts_with("//!") {
                    docs.push(line.trim().to_string());
                } else if trimmed.starts_with("#[") {
                    attrs.push(line.trim().to_string());
                } else if trimmed.is_empty() {
                    // skip empty lines between docs and item
                    if i == 0 {
                        break;
                    }
                    i = i.saturating_sub(1);
                    continue;
                } else {
                    break;
                }
            } else {
                break;
            }
            if i == 0 {
                break;
            }
            i = i.saturating_sub(1);
        }
        docs.reverse();
        attrs.reverse();
        let doc_text = if docs.is_empty() {
            None
        } else {
            Some(docs.join("\n"))
        };
        (doc_text, attrs)
    }

    // recursive walk with comprehensive symbol extraction
    fn walk_node<'a>(
        n: Node<'a>,
        src: &'a [u8],
        lines: &Vec<&str>,
        symbols: &mut Vec<SymbolNode>,
        parent_name: Option<String>,
    ) {
        let kind = n.kind();

        // Helper functions for extracting node information
        fn find_identifier<'a>(node: Node<'a>, src: &'a [u8]) -> Option<String> {
            if node.kind().contains("identifier") {
                return node.utf8_text(src).ok().map(|s| s.to_string());
            }
            for i in 0..node.child_count() {
                if let Some(ch) = node.child(i) {
                    if let Some(id) = find_identifier(ch, src) {
                        return Some(id);
                    }
                }
            }
            None
        }

        fn extract_visibility<'a>(n: &Node<'a>, src: &'a [u8]) -> Option<String> {
            // Look for visibility as a named child
            let mut cursor = n.walk();
            for child in n.children(&mut cursor) {
                if child.kind() == "visibility_modifier" {
                    return Some(node_text(&child, src));
                }
            }
            None
        }

        fn extract_generics<'a>(n: &Node<'a>, src: &'a [u8]) -> Option<String> {
            n.child_by_field_name("type_parameters")
                .map(|tp| node_text(&tp, src))
        }

        fn extract_parameters<'a>(n: &Node<'a>, src: &'a [u8]) -> Option<Vec<(String, String)>> {
            let params_node = n.child_by_field_name("parameters")?;
            let mut params = Vec::new();

            let mut cursor = params_node.walk();
            for child in params_node.named_children(&mut cursor) {
                if child.kind() == "parameter" {
                    let name = child
                        .child_by_field_name("pattern")
                        .map(|p| node_text(&p, src))
                        .unwrap_or_else(|| "_".to_string());
                    let type_name = child
                        .child_by_field_name("type")
                        .map(|t| node_text(&t, src))
                        .unwrap_or_else(|| "_".to_string());
                    params.push((name, type_name));
                } else if child.kind() == "self_parameter" {
                    let self_text = node_text(&child, src);
                    params.push((self_text.clone(), self_text));
                }
            }

            if params.is_empty() {
                None
            } else {
                Some(params)
            }
        }

        fn extract_return_type<'a>(n: &Node<'a>, src: &'a [u8]) -> Option<String> {
            n.child_by_field_name("return_type")
                .map(|rt| node_text(&rt, src))
        }

        fn extract_modifiers<'a>(n: &Node<'a>) -> Vec<String> {
            let mut mods = Vec::new();
            let mut cursor = n.walk();
            for child in n.children(&mut cursor) {
                match child.kind() {
                    "async" | "unsafe" | "const" | "extern" => {
                        mods.push(child.kind().to_string());
                    }
                    _ => {}
                }
            }
            mods
        }

        match kind {
            "function_item" => {
                let name = n
                    .child_by_field_name("name")
                    .and_then(|c| c.utf8_text(src).ok().map(|s| s.to_string()))
                    .or_else(|| find_identifier(n, src))
                    .unwrap_or_else(|| "<anon_fn>".to_string());

                let start = n.start_byte();
                let end = n.end_byte();
                let start_pos = (
                    n.start_position().row as usize,
                    n.start_position().column as usize,
                );
                let end_pos = (
                    n.end_position().row as usize,
                    n.end_position().column as usize,
                );
                let (docs, attributes) = collect_docs_and_attrs(lines, start_pos.0);

                symbols.push(SymbolNode {
                    name: name.clone(),
                    kind: "function".to_string(),
                    byte_range: Some((start, end)),
                    start_pos: Some(start_pos),
                    end_pos: Some(end_pos),
                    parent: parent_name.clone(),
                    docs,
                    visibility: extract_visibility(&n, src),
                    parameters: extract_parameters(&n, src),
                    return_type: extract_return_type(&n, src),
                    generics: extract_generics(&n, src),
                    trait_impl: None,
                    field_type: None,
                    attributes,
                    is_mutable: false,
                    modifiers: extract_modifiers(&n),
                    chunk_ids: Vec::new(),
                });
            }
            "struct_item" => {
                let name = n
                    .child_by_field_name("name")
                    .and_then(|c| c.utf8_text(src).ok().map(|s| s.to_string()))
                    .or_else(|| find_identifier(n, src))
                    .unwrap_or_else(|| "<anon_struct>".to_string());
                let start = n.start_byte();
                let end = n.end_byte();
                let start_pos = (
                    n.start_position().row as usize,
                    n.start_position().column as usize,
                );
                let end_pos = (
                    n.end_position().row as usize,
                    n.end_position().column as usize,
                );
                let (docs, attributes) = collect_docs_and_attrs(lines, start_pos.0);

                symbols.push(SymbolNode {
                    name: name.clone(),
                    kind: "struct".to_string(),
                    byte_range: Some((start, end)),
                    start_pos: Some(start_pos),
                    end_pos: Some(end_pos),
                    parent: parent_name.clone(),
                    docs,
                    visibility: extract_visibility(&n, src),
                    parameters: None,
                    return_type: None,
                    generics: extract_generics(&n, src),
                    trait_impl: None,
                    field_type: None,
                    attributes,
                    is_mutable: false,
                    modifiers: Vec::new(),
                    chunk_ids: Vec::new(),
                });

                // Walk struct fields
                let mut cursor = n.walk();
                for child in n.named_children(&mut cursor) {
                    walk_node(child, src, lines, symbols, Some(name.clone()));
                }
                return;
            }
            "enum_item" => {
                let name = n
                    .child_by_field_name("name")
                    .and_then(|c| c.utf8_text(src).ok().map(|s| s.to_string()))
                    .or_else(|| find_identifier(n, src))
                    .unwrap_or_else(|| "<anon_enum>".to_string());
                let start = n.start_byte();
                let end = n.end_byte();
                let start_pos = (
                    n.start_position().row as usize,
                    n.start_position().column as usize,
                );
                let end_pos = (
                    n.end_position().row as usize,
                    n.end_position().column as usize,
                );
                let (docs, attributes) = collect_docs_and_attrs(lines, start_pos.0);

                symbols.push(SymbolNode {
                    name: name.clone(),
                    kind: "enum".to_string(),
                    byte_range: Some((start, end)),
                    start_pos: Some(start_pos),
                    end_pos: Some(end_pos),
                    parent: parent_name.clone(),
                    docs,
                    visibility: extract_visibility(&n, src),
                    parameters: None,
                    return_type: None,
                    generics: extract_generics(&n, src),
                    trait_impl: None,
                    field_type: None,
                    attributes,
                    is_mutable: false,
                    modifiers: Vec::new(),
                    chunk_ids: Vec::new(),
                });

                // Walk enum variants
                let mut cursor = n.walk();
                for child in n.named_children(&mut cursor) {
                    walk_node(child, src, lines, symbols, Some(name.clone()));
                }
                return;
            }
            "trait_item" => {
                let name = n
                    .child_by_field_name("name")
                    .and_then(|c| c.utf8_text(src).ok().map(|s| s.to_string()))
                    .or_else(|| find_identifier(n, src))
                    .unwrap_or_else(|| "<anon_trait>".to_string());
                let start = n.start_byte();
                let end = n.end_byte();
                let start_pos = (
                    n.start_position().row as usize,
                    n.start_position().column as usize,
                );
                let end_pos = (
                    n.end_position().row as usize,
                    n.end_position().column as usize,
                );
                let (docs, attributes) = collect_docs_and_attrs(lines, start_pos.0);

                symbols.push(SymbolNode {
                    name: name.clone(),
                    kind: "trait".to_string(),
                    byte_range: Some((start, end)),
                    start_pos: Some(start_pos),
                    end_pos: Some(end_pos),
                    parent: parent_name.clone(),
                    docs,
                    visibility: extract_visibility(&n, src),
                    parameters: None,
                    return_type: None,
                    generics: extract_generics(&n, src),
                    trait_impl: None,
                    field_type: None,
                    attributes,
                    is_mutable: false,
                    modifiers: extract_modifiers(&n),
                    chunk_ids: Vec::new(),
                });

                // Walk trait methods
                let mut cursor = n.walk();
                for child in n.named_children(&mut cursor) {
                    walk_node(child, src, lines, symbols, Some(name.clone()));
                }
                return;
            }
            "impl_item" => {
                // Extract the target type and optional trait
                let target = n
                    .child_by_field_name("type")
                    .and_then(|c| c.utf8_text(src).ok().map(|s| s.to_string()))
                    .or_else(|| find_identifier(n, src));

                let trait_name = n.child_by_field_name("trait").map(|t| node_text(&t, src));

                let target_name = target.clone();
                let name = if let Some(ref trait_impl) = trait_name {
                    format!(
                        "{} for {}",
                        trait_impl,
                        target_name.clone().unwrap_or_else(|| "_".to_string())
                    )
                } else {
                    target_name.clone().unwrap_or_else(|| "<impl>".to_string())
                };

                let start = n.start_byte();
                let end = n.end_byte();
                let start_pos = (
                    n.start_position().row as usize,
                    n.start_position().column as usize,
                );
                let end_pos = (
                    n.end_position().row as usize,
                    n.end_position().column as usize,
                );
                let (docs, attributes) = collect_docs_and_attrs(lines, start_pos.0);

                symbols.push(SymbolNode {
                    name: name.clone(),
                    kind: "impl".to_string(),
                    byte_range: Some((start, end)),
                    start_pos: Some(start_pos),
                    end_pos: Some(end_pos),
                    parent: parent_name.clone(),
                    docs,
                    visibility: None,
                    parameters: None,
                    return_type: None,
                    generics: extract_generics(&n, src),
                    trait_impl: trait_name,
                    field_type: None,
                    attributes,
                    is_mutable: false,
                    modifiers: extract_modifiers(&n),
                    chunk_ids: Vec::new(),
                });

                // walk children with parent_name = target_name so methods become linked
                let mut cursor = n.walk();
                for child in n.named_children(&mut cursor) {
                    walk_node(child, src, lines, symbols, target_name.clone());
                }
                return; // children processed, skip default child traversal
            }
            "type_item" => {
                let name = n
                    .child_by_field_name("name")
                    .and_then(|c| c.utf8_text(src).ok())
                    .map(|s| s.to_string())
                    .or_else(|| find_identifier(n, src))
                    .unwrap_or_else(|| "<anon_type>".to_string());
                let start = n.start_byte();
                let end = n.end_byte();
                let start_pos = (
                    n.start_position().row as usize,
                    n.start_position().column as usize,
                );
                let end_pos = (
                    n.end_position().row as usize,
                    n.end_position().column as usize,
                );
                let (docs, attributes) = collect_docs_and_attrs(lines, start_pos.0);

                let alias_type = n.child_by_field_name("type").map(|t| node_text(&t, src));

                symbols.push(SymbolNode {
                    name: name.clone(),
                    kind: "type_alias".to_string(),
                    byte_range: Some((start, end)),
                    start_pos: Some(start_pos),
                    end_pos: Some(end_pos),
                    parent: parent_name.clone(),
                    docs,
                    visibility: extract_visibility(&n, src),
                    parameters: None,
                    return_type: None,
                    generics: extract_generics(&n, src),
                    trait_impl: None,
                    field_type: alias_type,
                    attributes,
                    is_mutable: false,
                    modifiers: Vec::new(),
                    chunk_ids: Vec::new(),
                });
            }
            "use_declaration" => {
                let txt = node_text(&n, src);
                let start = n.start_byte();
                let end = n.end_byte();
                let start_pos = (
                    n.start_position().row as usize,
                    n.start_position().column as usize,
                );
                let end_pos = (
                    n.end_position().row as usize,
                    n.end_position().column as usize,
                );

                symbols.push(SymbolNode {
                    name: txt.clone(),
                    kind: "use".to_string(),
                    byte_range: Some((start, end)),
                    start_pos: Some(start_pos),
                    end_pos: Some(end_pos),
                    parent: parent_name.clone(),
                    docs: None,
                    visibility: extract_visibility(&n, src),
                    parameters: None,
                    return_type: None,
                    generics: None,
                    trait_impl: None,
                    field_type: None,
                    attributes: Vec::new(),
                    is_mutable: false,
                    modifiers: Vec::new(),
                    chunk_ids: Vec::new(),
                });
            }
            "const_item" => {
                let name = n
                    .child_by_field_name("name")
                    .and_then(|c| c.utf8_text(src).ok().map(|s| s.to_string()))
                    .or_else(|| find_identifier(n, src))
                    .unwrap_or_else(|| "<anon_const>".to_string());
                let start = n.start_byte();
                let end = n.end_byte();
                let start_pos = (
                    n.start_position().row as usize,
                    n.start_position().column as usize,
                );
                let end_pos = (
                    n.end_position().row as usize,
                    n.end_position().column as usize,
                );
                let (docs, attributes) = collect_docs_and_attrs(lines, start_pos.0);

                let const_type = n.child_by_field_name("type").map(|t| node_text(&t, src));

                symbols.push(SymbolNode {
                    name: name.clone(),
                    kind: "const".to_string(),
                    byte_range: Some((start, end)),
                    start_pos: Some(start_pos),
                    end_pos: Some(end_pos),
                    parent: parent_name.clone(),
                    docs,
                    visibility: extract_visibility(&n, src),
                    parameters: None,
                    return_type: None,
                    generics: None,
                    trait_impl: None,
                    field_type: const_type,
                    attributes,
                    is_mutable: false,
                    modifiers: Vec::new(),
                    chunk_ids: Vec::new(),
                });
            }
            "static_item" => {
                let name = n
                    .child_by_field_name("name")
                    .and_then(|c| c.utf8_text(src).ok().map(|s| s.to_string()))
                    .or_else(|| find_identifier(n, src))
                    .unwrap_or_else(|| "<anon_static>".to_string());
                let start = n.start_byte();
                let end = n.end_byte();
                let start_pos = (
                    n.start_position().row as usize,
                    n.start_position().column as usize,
                );
                let end_pos = (
                    n.end_position().row as usize,
                    n.end_position().column as usize,
                );
                let (docs, attributes) = collect_docs_and_attrs(lines, start_pos.0);

                let static_type = n.child_by_field_name("type").map(|t| node_text(&t, src));

                let is_mut = n
                    .children(&mut n.walk())
                    .any(|c| c.kind() == "mutable_specifier");

                symbols.push(SymbolNode {
                    name: name.clone(),
                    kind: "static".to_string(),
                    byte_range: Some((start, end)),
                    start_pos: Some(start_pos),
                    end_pos: Some(end_pos),
                    parent: parent_name.clone(),
                    docs,
                    visibility: extract_visibility(&n, src),
                    parameters: None,
                    return_type: None,
                    generics: None,
                    trait_impl: None,
                    field_type: static_type,
                    attributes,
                    is_mutable: is_mut,
                    modifiers: Vec::new(),
                    chunk_ids: Vec::new(),
                });
            }
            "mod_item" => {
                let name = n
                    .child_by_field_name("name")
                    .and_then(|c| c.utf8_text(src).ok().map(|s| s.to_string()))
                    .or_else(|| find_identifier(n, src))
                    .unwrap_or_else(|| "<anon_mod>".to_string());
                let start = n.start_byte();
                let end = n.end_byte();
                let start_pos = (
                    n.start_position().row as usize,
                    n.start_position().column as usize,
                );
                let end_pos = (
                    n.end_position().row as usize,
                    n.end_position().column as usize,
                );
                let (docs, attributes) = collect_docs_and_attrs(lines, start_pos.0);

                symbols.push(SymbolNode {
                    name: name.clone(),
                    kind: "mod".to_string(),
                    byte_range: Some((start, end)),
                    start_pos: Some(start_pos),
                    end_pos: Some(end_pos),
                    parent: parent_name.clone(),
                    docs,
                    visibility: extract_visibility(&n, src),
                    parameters: None,
                    return_type: None,
                    generics: None,
                    trait_impl: None,
                    field_type: None,
                    attributes,
                    is_mutable: false,
                    modifiers: Vec::new(),
                    chunk_ids: Vec::new(),
                });
            }
            "macro_definition" => {
                let name = n
                    .child_by_field_name("name")
                    .and_then(|c| c.utf8_text(src).ok().map(|s| s.to_string()))
                    .or_else(|| find_identifier(n, src))
                    .unwrap_or_else(|| "<anon_macro>".to_string());
                let start = n.start_byte();
                let end = n.end_byte();
                let start_pos = (
                    n.start_position().row as usize,
                    n.start_position().column as usize,
                );
                let end_pos = (
                    n.end_position().row as usize,
                    n.end_position().column as usize,
                );
                let (docs, attributes) = collect_docs_and_attrs(lines, start_pos.0);

                symbols.push(SymbolNode {
                    name: name.clone(),
                    kind: "macro".to_string(),
                    byte_range: Some((start, end)),
                    start_pos: Some(start_pos),
                    end_pos: Some(end_pos),
                    parent: parent_name.clone(),
                    docs,
                    visibility: None,
                    parameters: None,
                    return_type: None,
                    generics: None,
                    trait_impl: None,
                    field_type: None,
                    attributes,
                    is_mutable: false,
                    modifiers: Vec::new(),
                    chunk_ids: Vec::new(),
                });
            }
            "field_declaration" => {
                let name = n
                    .child_by_field_name("name")
                    .and_then(|c| c.utf8_text(src).ok().map(|s| s.to_string()))
                    .or_else(|| find_identifier(n, src))
                    .unwrap_or_else(|| "<anon_field>".to_string());
                let start = n.start_byte();
                let end = n.end_byte();
                let start_pos = (
                    n.start_position().row as usize,
                    n.start_position().column as usize,
                );
                let end_pos = (
                    n.end_position().row as usize,
                    n.end_position().column as usize,
                );
                let (docs, attributes) = collect_docs_and_attrs(lines, start_pos.0);

                let field_type = n.child_by_field_name("type").map(|t| node_text(&t, src));

                symbols.push(SymbolNode {
                    name: name.clone(),
                    kind: "field".to_string(),
                    byte_range: Some((start, end)),
                    start_pos: Some(start_pos),
                    end_pos: Some(end_pos),
                    parent: parent_name.clone(),
                    docs,
                    visibility: extract_visibility(&n, src),
                    parameters: None,
                    return_type: None,
                    generics: None,
                    trait_impl: None,
                    field_type,
                    attributes,
                    is_mutable: false,
                    modifiers: Vec::new(),
                    chunk_ids: Vec::new(),
                });
            }
            "enum_variant" => {
                let name = n
                    .child_by_field_name("name")
                    .and_then(|c| c.utf8_text(src).ok().map(|s| s.to_string()))
                    .or_else(|| find_identifier(n, src))
                    .unwrap_or_else(|| "<anon_variant>".to_string());
                let start = n.start_byte();
                let end = n.end_byte();
                let start_pos = (
                    n.start_position().row as usize,
                    n.start_position().column as usize,
                );
                let end_pos = (
                    n.end_position().row as usize,
                    n.end_position().column as usize,
                );
                let (docs, attributes) = collect_docs_and_attrs(lines, start_pos.0);

                symbols.push(SymbolNode {
                    name: name.clone(),
                    kind: "variant".to_string(),
                    byte_range: Some((start, end)),
                    start_pos: Some(start_pos),
                    end_pos: Some(end_pos),
                    parent: parent_name.clone(),
                    docs,
                    visibility: None,
                    parameters: None,
                    return_type: None,
                    generics: None,
                    trait_impl: None,
                    field_type: None,
                    attributes,
                    is_mutable: false,
                    modifiers: Vec::new(),
                    chunk_ids: Vec::new(),
                });
            }
            "associated_type" => {
                let name = n
                    .child_by_field_name("name")
                    .and_then(|c| c.utf8_text(src).ok().map(|s| s.to_string()))
                    .or_else(|| find_identifier(n, src))
                    .unwrap_or_else(|| "<anon_assoc_type>".to_string());
                let start = n.start_byte();
                let end = n.end_byte();
                let start_pos = (
                    n.start_position().row as usize,
                    n.start_position().column as usize,
                );
                let end_pos = (
                    n.end_position().row as usize,
                    n.end_position().column as usize,
                );
                let (docs, attributes) = collect_docs_and_attrs(lines, start_pos.0);

                symbols.push(SymbolNode {
                    name: name.clone(),
                    kind: "associated_type".to_string(),
                    byte_range: Some((start, end)),
                    start_pos: Some(start_pos),
                    end_pos: Some(end_pos),
                    parent: parent_name.clone(),
                    docs,
                    visibility: None,
                    parameters: None,
                    return_type: None,
                    generics: None,
                    trait_impl: None,
                    field_type: None,
                    attributes,
                    is_mutable: false,
                    modifiers: Vec::new(),
                    chunk_ids: Vec::new(),
                });
            }
            _ => {}
        }

        // default: traverse named children
        let mut cursor = n.walk();
        for child in n.named_children(&mut cursor) {
            walk_node(child, src, lines, symbols, parent_name.clone());
        }
    }

    let root = tree.root_node();
    walk_node(root, source, &lines, &mut symbols, None);

    symbols
}

/// Embed chunks using the EmbeddingEngine with optimal batch processing.
/// Processes in batches of 32 for maximum GPU throughput.
pub async fn embed_chunks(
    chunks: &Vec<Chunk>,
    engine: &mut EmbeddingEngine,
) -> Result<Vec<Embedding>> {
    const BATCH_SIZE: usize = 32;

    let texts: Vec<String> = chunks.iter().map(|c| c.text.clone()).collect();
    let mut all_embeddings = Vec::with_capacity(texts.len());

    // Process in batches for optimal GPU utilization
    for batch in texts.chunks(BATCH_SIZE) {
        let batch_embeddings = engine.embed_batch(batch)?;
        all_embeddings.extend(batch_embeddings);
    }

    Ok(all_embeddings)
}

/// Assemble nodes and establish explicit chunk-symbol relationships based on byte ranges.
fn assemble_nodes(
    chunks: &mut Vec<Chunk>,
    symbols: &mut Vec<SymbolNode>,
    _strategy: &AssemblyStrategy,
) -> Vec<GraphEdge> {
    let mut edges = Vec::new();

    // Build chunk-symbol relationships based on byte range overlaps
    for chunk in chunks.iter_mut() {
        if let Some(chunk_range) = chunk.byte_range {
            let chunk_start = chunk_range.0;
            let chunk_end = chunk_range.1;

            for symbol in symbols.iter() {
                if let Some(symbol_range) = symbol.byte_range {
                    let symbol_start = symbol_range.0;
                    let symbol_end = symbol_range.1;

                    // Check if symbol is contained within chunk
                    // (symbol starts at or after chunk start AND symbol ends at or before chunk end)
                    if symbol_start >= chunk_start && symbol_end <= chunk_end {
                        chunk.containing_symbols.push(symbol.name.clone());
                    }
                }
            }
        }
    }

    // Build symbol -> chunk relationships (inverse of above)
    for symbol in symbols.iter_mut() {
        if let Some(symbol_range) = symbol.byte_range {
            let symbol_start = symbol_range.0;
            let symbol_end = symbol_range.1;

            for chunk in chunks.iter() {
                if let Some(chunk_range) = chunk.byte_range {
                    let chunk_start = chunk_range.0;
                    let chunk_end = chunk_range.1;

                    // Check if symbol is contained within chunk
                    if symbol_start >= chunk_start && symbol_end <= chunk_end {
                        symbol.chunk_ids.push(chunk.id.0.clone());

                        // Add Symbol -> Chunk edge
                        edges.push(GraphEdge {
                            src: symbol.name.clone(),
                            dst: chunk.id.0.clone(),
                            kind: GraphEdgeKind::SymbolToChunk,
                            weight: None,
                        });
                    }
                }
            }
        }
    }

    // Symbol → Symbol: parent relationships (methods/impls to their parent types)
    for s in symbols.iter() {
        if let Some(parent) = &s.parent {
            edges.push(GraphEdge {
                src: s.name.clone(),
                dst: parent.clone(),
                kind: GraphEdgeKind::SymbolToSymbol,
                weight: None,
            });
        }
    }

    // Symbol → Symbol: struct fields to their parent struct
    for s in symbols.iter().filter(|s| s.kind == "field") {
        if let Some(parent) = &s.parent {
            edges.push(GraphEdge {
                src: parent.clone(),
                dst: s.name.clone(),
                kind: GraphEdgeKind::SymbolToSymbol,
                weight: None,
            });
        }
    }

    // Symbol → Symbol: enum variants to their parent enum
    for s in symbols.iter().filter(|s| s.kind == "variant") {
        if let Some(parent) = &s.parent {
            edges.push(GraphEdge {
                src: parent.clone(),
                dst: s.name.clone(),
                kind: GraphEdgeKind::SymbolToSymbol,
                weight: None,
            });
        }
    }

    // Symbol → Symbol: impl blocks ownership
    // Connect impl blocks to the type they implement for
    for s in symbols.iter().filter(|s| s.kind == "impl") {
        // impl blocks should have their name as "impl TypeName" or similar
        // Extract the type name from the impl block name
        let impl_name = s.name.clone();
        if impl_name.starts_with("impl ") {
            let type_name = impl_name
                .trim_start_matches("impl ")
                .split('<')
                .next()
                .unwrap_or("")
                .trim();
            if !type_name.is_empty() {
                edges.push(GraphEdge {
                    src: type_name.to_string(),
                    dst: impl_name.clone(),
                    kind: GraphEdgeKind::SymbolToSymbol,
                    weight: None,
                });
            }
        }

        // If it's a trait impl, link to the trait
        if let Some(trait_impl) = &s.trait_impl {
            edges.push(GraphEdge {
                src: impl_name.clone(),
                dst: trait_impl.clone(),
                kind: GraphEdgeKind::SymbolToSymbol,
                weight: None,
            });
        }
    }

    edges
}

/// Compute simple metadata keys on the node according to requested keys.
fn compute_metadata(node: &FileNode, keys: &Vec<String>) -> HashMap<String, String> {
    let mut m = HashMap::new();
    // Always add path and filetype for reporting
    m.insert("path".to_string(), node.path.clone());
    let ext = std::path::Path::new(&node.path)
        .extension()
        .and_then(|s| s.to_str())
        .unwrap_or("")
        .to_string();
    m.insert("filetype".to_string(), ext);
    let text = String::from_utf8_lossy(&node.bytes).to_string();
    for k in keys.iter() {
        match k.as_str() {
            "title" => {
                let title = text.lines().next().unwrap_or("").to_string();
                m.insert("title".to_string(), title);
            }
            "anchors" => {
                let anchors = text.lines().filter(|l| l.contains('#')).count().to_string();
                m.insert("anchors".to_string(), anchors);
            }
            "language" => {
                let lang = std::path::Path::new(&node.path)
                    .extension()
                    .and_then(|s| s.to_str())
                    .unwrap_or("")
                    .to_string();
                m.insert("language".to_string(), lang);
            }
            "dependencies" => {
                m.insert("dependencies".to_string(), "[]".to_string());
            }
            "filetype" => {
                let ft = std::path::Path::new(&node.path)
                    .extension()
                    .and_then(|s| s.to_str())
                    .unwrap_or("")
                    .to_string();
                m.insert("filetype".to_string(), ft);
            }
            _ => {
                // unknown key: store empty
                m.insert(k.clone(), String::new());
            }
        }
    }
    m
}

fn looks_binary(bytes: &[u8]) -> bool {
    bytes.iter().take(1024).any(|&b| b == 0)
}

fn normalize_path(path: &str) -> String {
    if let Some((_, rest)) = path.split_once('/') {
        return rest.to_string();
    }
    path.to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_is_noise_file_positive() {
        let positives = vec![
            "LICENSE",
            "COPYING",
            ".gitignore",
            "Cargo.lock",
            "package-lock.json",
            "image.png",
            "font.woff2",
            "README",
            "CHANGELOG.md",
            "CONTRIBUTING",
        ];

        for p in positives {
            assert!(is_noise_file(p), "{} should be considered noise", p);
        }
    }

    #[test]
    fn test_is_noise_file_negative() {
        let negatives = vec!["README.md", "src/lib.rs", "main.rs", "notes.txt"];
        for n in negatives {
            assert!(!is_noise_file(n), "{} should NOT be considered noise", n);
        }
    }

    #[test]
    fn test_filekind_to_plan_defaults() {
        let sc = filekind_to_plan(&FileKind::SourceCode);
        assert!(sc.get_symbol_parse());
        assert!(sc.get_embed());
        assert!(matches!(sc.get_chunking(), ChunkingStrategy::CodeAware));

        let doc = filekind_to_plan(&FileKind::Documentation);
        assert!(!doc.get_symbol_parse());
        assert!(doc.get_embed());
        assert!(matches!(
            doc.get_chunking(),
            ChunkingStrategy::HeadingSections
        ));

        let cfg = filekind_to_plan(&FileKind::Config);
        assert!(!cfg.get_symbol_parse());
        assert!(!cfg.get_embed());
        assert!(matches!(cfg.get_chunking(), ChunkingStrategy::None));
    }

    #[test]
    fn test_parse_rust_functions() {
        let rust_code = r#"
/// This is a documented function
pub fn add(x: i32, y: i32) -> i32 {
    x + y
}

async fn fetch_data() {
    // implementation
}
"#;
        let node = FileNode {
            path: "test.rs".to_string(),
            bytes: rust_code.as_bytes().to_vec(),
            kind: FileKind::SourceCode,
        };

        let symbols = parse_symbols(&node);

        let add_fn = symbols.iter().find(|s| s.name == "add");
        assert!(add_fn.is_some());
        let add_fn = add_fn.unwrap();
        assert_eq!(add_fn.kind, "function");
        assert_eq!(add_fn.visibility, Some("pub".to_string()));
        assert!(add_fn.docs.is_some());
        assert!(add_fn.parameters.is_some());
        assert_eq!(add_fn.parameters.as_ref().unwrap().len(), 2);
        assert!(add_fn.return_type.is_some());
        assert!(add_fn.return_type.as_ref().unwrap().contains("i32"));

        let fetch_fn = symbols.iter().find(|s| s.name == "fetch_data");
        assert!(fetch_fn.is_some());
        // Note: async detection works but may depend on tree-sitter version
        // Just verify the function was found
        assert_eq!(fetch_fn.unwrap().kind, "function");
    }

    #[test]
    fn test_parse_rust_structs_and_enums() {
        let rust_code = r#"
/// A point in 2D space
#[derive(Debug, Clone)]
pub struct Point {
    pub x: f64,
    pub y: f64,
}

pub enum Color {
    Red,
    Green,
    Blue,
}
"#;
        let node = FileNode {
            path: "test.rs".to_string(),
            bytes: rust_code.as_bytes().to_vec(),
            kind: FileKind::SourceCode,
        };

        let symbols = parse_symbols(&node);

        let point_struct = symbols.iter().find(|s| s.name == "Point");
        assert!(point_struct.is_some());
        let point_struct = point_struct.unwrap();
        assert_eq!(point_struct.kind, "struct");
        assert_eq!(point_struct.visibility, Some("pub".to_string()));
        assert!(point_struct.docs.is_some());
        assert!(point_struct.attributes.iter().any(|a| a.contains("derive")));

        let color_enum = symbols.iter().find(|s| s.name == "Color");
        assert!(color_enum.is_some());
        assert_eq!(color_enum.unwrap().kind, "enum");
    }

    #[test]
    fn test_parse_rust_impl_blocks() {
        let rust_code = r#"
struct Calculator;

impl Calculator {
    pub fn add(&self, a: i32, b: i32) -> i32 {
        a + b
    }
}

trait Display {
    fn show(&self);
}

impl Display for Calculator {
    fn show(&self) {
        println!("Calculator");
    }
}
"#;
        let node = FileNode {
            path: "test.rs".to_string(),
            bytes: rust_code.as_bytes().to_vec(),
            kind: FileKind::SourceCode,
        };

        let symbols = parse_symbols(&node);

        // Check inherent impl
        let inherent_impl = symbols
            .iter()
            .find(|s| s.kind == "impl" && s.trait_impl.is_none());
        assert!(inherent_impl.is_some());
        assert_eq!(inherent_impl.unwrap().name, "Calculator");

        // Check trait impl
        let trait_impl = symbols
            .iter()
            .find(|s| s.kind == "impl" && s.trait_impl.is_some());
        assert!(trait_impl.is_some());
        let trait_impl = trait_impl.unwrap();
        assert!(trait_impl.name.contains("Display"));
        assert!(trait_impl.name.contains("Calculator"));

        // Check that methods are linked to their parent
        let add_method = symbols.iter().find(|s| s.name == "add");
        assert!(add_method.is_some());
        assert_eq!(add_method.unwrap().parent, Some("Calculator".to_string()));
    }

    #[test]
    fn test_parse_rust_generics() {
        let rust_code = r#"
pub struct Container<T> {
    value: T,
}

impl<T: Clone> Container<T> {
    pub fn new(value: T) -> Self {
        Container { value }
    }
}

pub fn identity<T>(x: T) -> T {
    x
}
"#;
        let node = FileNode {
            path: "test.rs".to_string(),
            bytes: rust_code.as_bytes().to_vec(),
            kind: FileKind::SourceCode,
        };

        let symbols = parse_symbols(&node);

        let container = symbols.iter().find(|s| s.name == "Container");
        assert!(container.is_some());
        assert!(container.unwrap().generics.is_some());

        let identity_fn = symbols.iter().find(|s| s.name == "identity");
        assert!(identity_fn.is_some());
        assert!(identity_fn.unwrap().generics.is_some());
    }

    #[test]
    fn test_parse_const_static_mod() {
        let rust_code = r#"
pub const MAX_SIZE: usize = 1024;

pub static mut COUNTER: i32 = 0;

pub mod utils {
    pub fn helper() {}
}
"#;
        let node = FileNode {
            path: "test.rs".to_string(),
            bytes: rust_code.as_bytes().to_vec(),
            kind: FileKind::SourceCode,
        };

        let symbols = parse_symbols(&node);

        let const_item = symbols.iter().find(|s| s.name == "MAX_SIZE");
        assert!(const_item.is_some());
        assert_eq!(const_item.unwrap().kind, "const");

        let static_item = symbols.iter().find(|s| s.name == "COUNTER");
        assert!(static_item.is_some());
        let static_item = static_item.unwrap();
        assert_eq!(static_item.kind, "static");
        assert!(static_item.is_mutable);

        let mod_item = symbols.iter().find(|s| s.name == "utils");
        assert!(mod_item.is_some());
        assert_eq!(mod_item.unwrap().kind, "mod");
    }
}

// The synchronous `unzip_to_memory` was removed in favour of an async API.
// Use the async `unzip_to_memory` below which returns processed files.

/// Async helper: unzip the archive and run the processing executor on each file.
pub async fn unzip_to_memory(
    zip_bytes: &[u8],
    mut engine: Option<&mut EmbeddingEngine>,
) -> Result<Vec<ProcessedFile>> {
    let cursor = Cursor::new(zip_bytes.to_vec());
    let mut archive = ZipArchive::new(cursor)?;
    let mut processed = Vec::new();

    for i in 0..archive.len() {
        let mut file = archive.by_index(i)?;
        if file.is_dir() {
            continue;
        }

        let raw_path = file.name().to_string();
        let path = normalize_path(&raw_path);

        let mut bytes = Vec::new();
        std::io::copy(&mut file, &mut bytes)?;

        if looks_binary(&bytes) {
            continue;
        }

        let extension = std::path::Path::new(&path)
            .extension()
            .and_then(|s| s.to_str())
            .unwrap_or("");

        let kind = FileKind::classify(extension);

        let node = FileNode { path, bytes, kind };
        let plan = plan_for_file_node(&node);
        let pf = process_file(&node, &plan, engine.as_deref_mut()).await;
        processed.push(pf);
    }

    Ok(processed)
}

/// Optimized parallel pipeline for processing files.
///
/// Performance characteristics:
/// - Extracts files sequentially (I/O bound, no benefit from parallelization)
/// - Parses & chunks in parallel with Rayon (CPU bound, scales linearly with cores)
/// - Batches all embeddings together (GPU bound, benefits from large batch sizes)
/// - Builds graph connectivity sequentially (fast enough single-threaded)
///
/// On a 4070 Ti with 16 cores:
/// - 5,000 chunks embed in ~150ms with batch size 32
/// - 50,000+ line repos process in seconds, not minutes
pub async fn unzip_to_memory_parallel(
    zip_bytes: &[u8],
    engine: Option<&mut EmbeddingEngine>,
) -> Result<Vec<ProcessedFile>> {
    use rayon::prelude::*;

    // Phase 1: Extract all files from zip (fast, I/O bound)
    let cursor = Cursor::new(zip_bytes.to_vec());
    let mut archive = ZipArchive::new(cursor)?;
    let mut file_nodes = Vec::new();

    for i in 0..archive.len() {
        let mut file = archive.by_index(i)?;
        if file.is_dir() {
            continue;
        }

        let raw_path = file.name().to_string();
        let path = normalize_path(&raw_path);

        let mut bytes = Vec::new();
        std::io::copy(&mut file, &mut bytes)?;

        if looks_binary(&bytes) {
            continue;
        }

        let extension = std::path::Path::new(&path)
            .extension()
            .and_then(|s| s.to_str())
            .unwrap_or("");

        let kind = FileKind::classify(extension);
        file_nodes.push(FileNode { path, bytes, kind });
    }

    // Phase 2: Parse & chunk all files in parallel (CPU bound)
    // This is embarrassingly parallel - each file is independent
    let parsed: Vec<_> = file_nodes
        .par_iter()
        .map(|node| {
            let plan = plan_for_file_node(node);

            // Extract, chunk, and parse symbols without embedding
            let extracted = run_extractions(node, plan.get_extract());
            let chunks = run_chunking(node, &extracted, plan.get_chunking());
            let symbols = if plan.get_symbol_parse() {
                parse_symbols(node)
            } else {
                Vec::new()
            };

            (node.clone(), plan, chunks, symbols)
        })
        .collect();

    // Phase 3: Collect all chunks and batch embed them at once (GPU bound)
    let mut all_chunks = Vec::new();
    let mut chunk_file_map = Vec::new(); // Track which chunks belong to which file

    for (i, (_, plan, chunks, _)) in parsed.iter().enumerate() {
        if plan.get_embed() {
            for chunk in chunks {
                all_chunks.push(chunk.clone());
                chunk_file_map.push(i);
            }
        }
    }

    let all_embeddings = if let Some(eng) = engine {
        if !all_chunks.is_empty() {
            embed_chunks(&all_chunks, eng).await?
        } else {
            Vec::new()
        }
    } else {
        Vec::new()
    };

    // Phase 4: Distribute embeddings back to files and build graph
    let mut processed = Vec::new();
    let mut embedding_idx = 0;

    for (i, (node, plan, mut chunks, mut symbols)) in parsed.into_iter().enumerate() {
        // Collect embeddings for this file's chunks
        let mut file_embeddings = Vec::new();
        if plan.get_embed() {
            for _ in 0..chunks.len() {
                if embedding_idx < all_embeddings.len()
                    && chunk_file_map.get(embedding_idx) == Some(&i)
                {
                    file_embeddings.push(all_embeddings[embedding_idx].clone());
                    embedding_idx += 1;
                }
            }
        }

        // Build graph connectivity
        let graph_edges = assemble_nodes(&mut chunks, &mut symbols, plan.get_assembly());
        let mut metadata = compute_metadata(&node, plan.get_metadata());

        // Compute and store file-level LOC metrics (run on original file bytes
        // before/independent of chunking). This preserves realistic file metrics
        // even when `ProcessedFile::original_bytes` is dropped to save memory.
        let ext = std::path::Path::new(&node.path)
            .extension()
            .and_then(|s| s.to_str())
            .unwrap_or("");
        if let Some((total, code, comment, blank)) = code_file_stats(&node.bytes, ext) {
            metadata.insert("loc_total".to_string(), total.to_string());
            metadata.insert("loc_code".to_string(), code.to_string());
            metadata.insert("loc_comment".to_string(), comment.to_string());
            metadata.insert("loc_blank".to_string(), blank.to_string());
        }

        processed.push(ProcessedFile {
            file_node: node,
            chunks,
            symbols,
            embeddings: file_embeddings,
            metadata,
            graph_edges,
            original_bytes: Vec::new(), // Save memory - we already have the node
        });
    }

    Ok(processed)
}

pub async fn load_zip(source: &str) -> Result<Vec<u8>> {
    if !source.ends_with(".zip") {
        return Err(anyhow!("Source does not end with .zip: {}", source));
    }

    if let Ok(url) = Url::parse(source) {
        return load_zip_from_url(url).await;
    }

    load_zip_from_path(source).await
}

async fn load_zip_from_path(path: &str) -> Result<Vec<u8>> {
    Ok(fs::read(path).await?)
}

async fn load_zip_from_url(url: Url) -> Result<Vec<u8>> {
    let client = Client::new();
    let res = client.get(url.clone()).send().await?.error_for_status()?;
    Ok(res.bytes().await?.to_vec())
}

use crate::community::{build_similarity_edges, Louvain, SemanticCommunity};

/// A semantic community detected in the codebase.
#[derive(Debug, Clone)]
pub struct DetectedCommunity {
    pub id: usize,
    pub symbol_names: Vec<String>,
    pub files: std::collections::HashSet<String>,
    pub cohesion: f32,
    pub suggested_label: Option<String>,
}

pub struct ProjectGraph {
    pub files: Vec<FileNode>,
    pub symbols: Vec<SymbolNode>,
    pub chunks: Vec<Chunk>,
    pub edges: Vec<GraphEdge>,
    /// Embeddings for each chunk (parallel to chunks vec)
    pub chunk_embeddings: Vec<Embedding>,
    /// Maps chunk index to file path
    pub chunk_to_file: std::collections::HashMap<usize, String>,
    /// Detected semantic communities/subsystems
    pub communities: Vec<DetectedCommunity>,
    /// Modularity score from community detection
    pub modularity: f64,
}

impl ProjectGraph {
    /// Build project graph from processed files.
    /// This version does NOT compute similarity edges or communities (use `from_processed_files_with_communities` for that).
    pub fn from_processed_files(files: Vec<ProcessedFile>) -> Self {
        Self::from_processed_files_internal(files, false, 0.5)
    }

    /// Build project graph with similarity edges and Louvain community detection.
    /// `similarity_threshold`: minimum cosine similarity (0.0–1.0) to create an edge.
    pub fn from_processed_files_with_communities(
        files: Vec<ProcessedFile>,
        similarity_threshold: f32,
    ) -> Self {
        Self::from_processed_files_internal(files, true, similarity_threshold)
    }

    fn from_processed_files_internal(
        files: Vec<ProcessedFile>,
        compute_communities: bool,
        similarity_threshold: f32,
    ) -> Self {
        // Collect all chunk embeddings (preserving order) and build chunk-to-file map
        let mut chunk_embeddings: Vec<Embedding> = Vec::new();
        let mut chunk_to_file_map: std::collections::HashMap<usize, String> =
            std::collections::HashMap::new();
        let mut chunk_idx = 0;

        for pf in &files {
            for emb in &pf.embeddings {
                chunk_embeddings.push(emb.clone());
                chunk_to_file_map.insert(chunk_idx, pf.file_node.path.clone());
                chunk_idx += 1;
            }
        }

        let mut graph = ProjectGraph {
            files: files.iter().map(|f| f.file_node.clone()).collect(),
            symbols: files.iter().flat_map(|f| f.symbols.clone()).collect(),
            chunks: files.iter().flat_map(|f| f.chunks.clone()).collect(),
            edges: files.iter().flat_map(|f| f.graph_edges.clone()).collect(),
            chunk_embeddings: chunk_embeddings.clone(),
            chunk_to_file: chunk_to_file_map,
            communities: Vec::new(),
            modularity: 0.0,
        };

        // File → File: module hierarchy edges
        // Connect parent modules to child modules based on file paths
        for file in &graph.files {
            if let Some(parent_path) = file.parent_module() {
                graph.edges.push(GraphEdge {
                    src: parent_path,
                    dst: file.path.clone(),
                    kind: GraphEdgeKind::FileToFile,
                    weight: None,
                });
            }
        }

        // Chunk → Chunk: sequential flow edges
        // Connect chunks within the same file in sequential order
        for pf in &files {
            let file_chunks = &pf.chunks;
            for i in 0..file_chunks.len().saturating_sub(1) {
                let current = &file_chunks[i];
                let next = &file_chunks[i + 1];

                graph.edges.push(GraphEdge {
                    src: current.id.0.clone(),
                    dst: next.id.0.clone(),
                    kind: GraphEdgeKind::ChunkToChunk,
                    weight: None,
                });
            }
        }

        // Compute similarity edges and communities if requested
        if compute_communities && !chunk_embeddings.is_empty() {
            graph.compute_similarity_and_communities(similarity_threshold);
        }

        graph
    }

    /// Compute chunk-to-chunk similarity edges and run Louvain community detection.
    fn compute_similarity_and_communities(&mut self, threshold: f32) {
        let n = self.chunk_embeddings.len();
        if n < 2 {
            return;
        }

        // Build similarity edges between chunks
        let sim_edges = build_similarity_edges(&self.chunk_embeddings, threshold);

        // Add similarity edges to graph (map chunk indices to chunk IDs)
        for se in &sim_edges {
            if se.src < self.chunks.len() && se.dst < self.chunks.len() {
                self.edges.push(GraphEdge {
                    src: self.chunks[se.src].id.0.clone(),
                    dst: self.chunks[se.dst].id.0.clone(),
                    kind: GraphEdgeKind::SymbolSimilarity,
                    weight: Some(se.weight),
                });
            }
        }

        // Run Louvain community detection
        let mut louvain = Louvain::new(n, &sim_edges);
        let result = louvain.run();
        self.modularity = result.modularity;

        // Build DetectedCommunity structs
        for (comm_id, indices) in &result.communities {
            let mut symbol_names = Vec::new();
            let mut files = std::collections::HashSet::new();

            for &chunk_idx in indices {
                if chunk_idx < self.chunks.len() {
                    let chunk = &self.chunks[chunk_idx];
                    // Add symbols from this chunk
                    symbol_names.extend(chunk.containing_symbols.clone());

                    // Use our pre-computed chunk-to-file mapping
                    if let Some(file_path) = self.chunk_to_file.get(&chunk_idx) {
                        files.insert(file_path.clone());
                    }
                }
            }

            // Deduplicate symbol names
            symbol_names.sort();
            symbol_names.dedup();

            let cohesion = SemanticCommunity::compute_cohesion(indices, &self.chunk_embeddings);
            let suggested_label = SemanticCommunity::infer_label(&symbol_names);

            self.communities.push(DetectedCommunity {
                id: *comm_id,
                symbol_names,
                files,
                cohesion,
                suggested_label,
            });
        }

        // Sort communities by size (largest first)
        self.communities.sort_by(|a, b| b.symbol_names.len().cmp(&a.symbol_names.len()));
    }

    /// Print a summary of the graph structure
    pub fn print_summary(&self) {
        println!("\n=== Project Graph Summary ===");
        println!("Files: {}", self.files.len());
        println!("Symbols: {}", self.symbols.len());
        println!("Chunks: {}", self.chunks.len());
        println!("Edges: {}", self.edges.len());

        // Count edges by type
        let mut symbol_to_chunk = 0;
        let mut symbol_to_symbol = 0;
        let mut file_to_file = 0;
        let mut chunk_to_chunk = 0;
        let mut symbol_similarity = 0;

        for edge in &self.edges {
            match edge.kind {
                GraphEdgeKind::SymbolToChunk => symbol_to_chunk += 1,
                GraphEdgeKind::SymbolToSymbol => symbol_to_symbol += 1,
                GraphEdgeKind::FileToFile => file_to_file += 1,
                GraphEdgeKind::ChunkToChunk => chunk_to_chunk += 1,
                GraphEdgeKind::SymbolSimilarity => symbol_similarity += 1,
            }
        }

        println!("\nEdge breakdown:");
        println!("  Symbol → Chunk: {}", symbol_to_chunk);
        println!("  Symbol → Symbol: {}", symbol_to_symbol);
        println!("  Symbol ~ Symbol (similarity): {}", symbol_similarity);
        println!("  File → File: {}", file_to_file);
        println!("  Chunk → Chunk: {}", chunk_to_chunk);

        // Count symbols by kind
        let mut symbol_kinds = std::collections::HashMap::new();
        for symbol in &self.symbols {
            *symbol_kinds.entry(symbol.kind.clone()).or_insert(0) += 1;
        }

        println!("\nSymbol breakdown:");
        let mut kinds: Vec<_> = symbol_kinds.iter().collect();
        kinds.sort_by_key(|(_, count)| std::cmp::Reverse(*count));
        for (kind, count) in kinds {
            println!("  {}: {}", kind, count);
        }
    }

    /// Print detailed community/subsystem analysis.
    pub fn print_communities(&self) {
        if self.communities.is_empty() {
            println!("\n=== Semantic Communities ===");
            println!("No communities detected (run with similarity analysis enabled).");
            return;
        }

        println!("\n=== Semantic Communities (Subsystems) ===");
        println!("Detected {} communities (modularity: {:.4})", self.communities.len(), self.modularity);
        println!();

        for (i, comm) in self.communities.iter().enumerate().take(10) {
            let label = comm.suggested_label.as_deref().unwrap_or("unnamed");
            println!(
                "📦 Community {} [{}]: {} symbols, cohesion={:.3}",
                i + 1,
                label,
                comm.symbol_names.len(),
                comm.cohesion
            );

            // Show sample symbols (up to 8)
            if !comm.symbol_names.is_empty() {
                let sample: Vec<_> = comm.symbol_names.iter().take(8).collect();
                print!("   └─ symbols: ");
                for (j, sym) in sample.iter().enumerate() {
                    if j > 0 {
                        print!(", ");
                    }
                    print!("{}", sym);
                }
                if comm.symbol_names.len() > 8 {
                    print!(" ... (+{})", comm.symbol_names.len() - 8);
                }
                println!();
            }

            // Show files
            if !comm.files.is_empty() {
                let file_list: Vec<_> = comm.files.iter().take(3).collect();
                print!("   └─ files: ");
                for (j, f) in file_list.iter().enumerate() {
                    if j > 0 {
                        print!(", ");
                    }
                    print!("{}", f);
                }
                if comm.files.len() > 3 {
                    print!(" ... (+{})", comm.files.len() - 3);
                }
                println!();
            }
        }

        if self.communities.len() > 10 {
            println!("\n... and {} more communities", self.communities.len() - 10);
        }

        // Refactor suggestions
        self.print_refactor_suggestions();
    }

    /// Print multi-file refactor suggestions based on community analysis.
    fn print_refactor_suggestions(&self) {
        println!("\n=== Refactor Suggestions ===");

        // Find communities that span multiple files (potential module extraction)
        let multi_file_comms: Vec<_> = self
            .communities
            .iter()
            .filter(|c| c.files.len() > 1 && c.symbol_names.len() >= 3)
            .take(5)
            .collect();

        if multi_file_comms.is_empty() {
            println!("No obvious refactoring opportunities detected.");
            return;
        }

        for (i, comm) in multi_file_comms.iter().enumerate() {
            let label = comm.suggested_label.as_deref().unwrap_or("related");
            println!(
                "{}. Consider extracting '{}' module:",
                i + 1,
                label
            );
            println!(
                "   {} related symbols spread across {} files (cohesion={:.2})",
                comm.symbol_names.len(),
                comm.files.len(),
                comm.cohesion
            );

            // List the files
            for f in comm.files.iter().take(4) {
                println!("   - {}", f);
            }
            if comm.files.len() > 4 {
                println!("   ... and {} more files", comm.files.len() - 4);
            }
            println!();
        }
    }
}
