use anyhow::{Result, anyhow};
use reqwest::Client;
use std::io::Cursor;
use tokio::fs;
use url::Url;
use zip::ZipArchive;

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
#[derive(Debug, Clone)]
pub enum ChunkingStrategy {
    Lines(usize),      // chunk by N lines
    Delimiter(String), // chunk by a delimiter string
    HeadingSections,   // chunk by markdown headings
    CodeAware,         // language-aware code chunking
    None,              // no chunking
}

/// How to assemble graph nodes from the extracted/chunked pieces.
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
    filekind_to_plan(&node.kind)
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

pub fn unzip_to_memory(zip_bytes: &[u8]) -> Result<Vec<FileNode>> {
    let cursor = Cursor::new(zip_bytes.to_vec());
    let mut archive = ZipArchive::new(cursor)?;
    let mut files = Vec::new();

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

        files.push(FileNode { path, bytes, kind });
    }

    Ok(files)
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
