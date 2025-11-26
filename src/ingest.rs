use anyhow::{Result, anyhow};
use reqwest::Client;
use std::collections::HashMap;
use std::io::Cursor;
use tokio::fs;
use url::Url;
use zip::ZipArchive;

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
}

/// A minimal symbol node representation parsed from source files.
#[allow(dead_code)]
#[derive(Debug, Clone)]
pub struct SymbolNode {
    pub name: String,
    pub kind: String,
    pub location: Option<(usize, usize)>,
}

/// Embedding vector placeholder.
#[allow(dead_code)]
pub type Embedding = Vec<f32>;

/// ProcessedFile is the executor output for a single file + plan.
#[derive(Debug, Clone)]
pub struct ProcessedFile {
    pub chunks: Vec<Chunk>,
    pub symbols: Vec<SymbolNode>,
    pub embeddings: Vec<Embedding>,
    pub metadata: HashMap<String, String>,
}

impl ProcessedFile {
    pub fn empty(_node: &FileNode) -> Self {
        ProcessedFile {
            chunks: Vec::new(),
            symbols: Vec::new(),
            embeddings: Vec::new(),
            metadata: HashMap::new(),
        }
    }
}

/// Orchestrator: execute a ProcessingPlan on a `FileNode`.
pub async fn process_file(node: &FileNode, plan: &ProcessingPlan) -> ProcessedFile {
    if plan.get_skip() {
        return ProcessedFile::empty(node);
    }

    // 1) extraction
    let extracted = run_extractions(node, plan.get_extract());

    // 2) chunking
    let chunks = run_chunking(node, &extracted, plan.get_chunking());

    // 3) symbols (optional)
    let symbols = if plan.get_symbol_parse() {
        parse_symbols(node)
    } else {
        Vec::new()
    };

    // 4) embeddings (optional, async to support remote embedders)
    let embeddings = if plan.get_embed() {
        embed_chunks(&chunks).await
    } else {
        Vec::new()
    };

    // 5) assembly
    let _assembled = assemble_nodes(&chunks, &symbols, plan.get_assembly());

    ProcessedFile {
        chunks,
        symbols,
        embeddings,
        metadata: compute_metadata(node, plan.get_metadata()),
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

/// Run chunking on the extracted pieces. Very small, heuristic implementation.
fn run_chunking(
    _node: &FileNode,
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
                    });
                }
            }
            if out.is_empty() {
                out.push(Chunk {
                    id: ChunkId("empty".to_string()),
                    text: String::new(),
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
                });
            }
            out
        }
        ChunkingStrategy::CodeAware => {
            // naive: split by blank line segments
            let full = extracted
                .iter()
                .map(|(_, t)| t.clone())
                .collect::<Vec<_>>()
                .join("\n");
            full.split("\n\n")
                .enumerate()
                .map(|(i, s)| Chunk {
                    id: ChunkId(format!("c-{}", i)),
                    text: s.trim().to_string(),
                })
                .collect()
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
                })
                .collect()
        }
    }
}

/// Parse symbols from the node. Placeholder implementation that returns no symbols for now.
fn parse_symbols(node: &FileNode) -> Vec<SymbolNode> {
    // As a simple heuristic: for Rust files, collect `fn ` names (very naive)
    let ext = std::path::Path::new(&node.path)
        .extension()
        .and_then(|s| s.to_str())
        .unwrap_or("");
    let text = String::from_utf8_lossy(&node.bytes).to_string();
    if ext == "rs" {
        text.lines()
            .filter_map(|l| {
                let t = l.trim_start();
                if t.starts_with("fn ") {
                    let rest = t.strip_prefix("fn ").unwrap_or("");
                    let name = rest
                        .split(|c: char| c == '(' || c.is_whitespace())
                        .next()
                        .unwrap_or("");
                    Some(SymbolNode {
                        name: name.to_string(),
                        kind: "function".to_string(),
                        location: None,
                    })
                } else {
                    None
                }
            })
            .collect()
    } else {
        Vec::new()
    }
}

/// Embed chunks. Placeholder that returns zero-length vectors.
async fn embed_chunks(_chunks: &Vec<Chunk>) -> Vec<Embedding> {
    // In production this would call an async embedding service.
    Vec::new()
}

/// Assemble nodes; placeholder returns nothing but could build edges in future.
fn assemble_nodes(
    _chunks: &Vec<Chunk>,
    _symbols: &Vec<SymbolNode>,
    _strategy: &AssemblyStrategy,
) -> Vec<()> {
    Vec::new()
}

/// Compute simple metadata keys on the node according to requested keys.
fn compute_metadata(node: &FileNode, keys: &Vec<String>) -> HashMap<String, String> {
    let mut m = HashMap::new();
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
}

// The synchronous `unzip_to_memory` was removed in favour of an async API.
// Use the async `unzip_to_memory` below which returns processed files.

/// Async helper: unzip the archive and run the processing executor on each file.
pub async fn unzip_to_memory(zip_bytes: &[u8]) -> Result<Vec<ProcessedFile>> {
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
        let pf = process_file(&node, &plan).await;
        processed.push(pf);
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
