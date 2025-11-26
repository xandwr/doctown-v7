// editor.rs - Read-only file content access and chunk retrieval

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::io::Read;
use zip::ZipArchive;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FileRequest {
    pub path: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FileResponse {
    pub path: String,
    pub content: String,
    pub size: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChunkRequest {
    pub chunk_id: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChunkResponse {
    pub chunk_id: String,
    pub file_path: String,
    pub text: String,
    pub start: usize,
    pub end: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SymbolContentRequest {
    pub symbol: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SymbolContentResponse {
    pub symbol: String,
    pub file: String,
    pub content: String,
    pub chunk_range: String,
}

/// Read a file from the docpack archive
pub fn read_file(
    request: &FileRequest,
    archive: &mut ZipArchive<std::fs::File>,
) -> Result<FileResponse> {
    // Validate path is not empty
    if request.path.is_empty() {
        anyhow::bail!("File path cannot be empty");
    }

    // Try to find the file in source/ directory
    let file_path = format!("source/{}", request.path.trim_start_matches('/'));

    let mut file = archive
        .by_name(&file_path)
        .context(format!("File not found in docpack: '{}' (looking for '{}')", request.path, file_path))?;

    let mut content = String::new();
    file.read_to_string(&mut content)
        .context(format!("Failed to read file content from '{}'", request.path))?;

    let size = content.len();

    Ok(FileResponse {
        path: request.path.clone(),
        content,
        size,
    })
}

/// Get a specific chunk by ID
pub fn get_chunk(
    request: &ChunkRequest,
    chunks: &[doctown::docpack::ChunkEntry],
) -> Result<Option<ChunkResponse>> {
    let chunk = chunks.iter().find(|c| c.chunk_id == request.chunk_id);

    if let Some(c) = chunk {
        Ok(Some(ChunkResponse {
            chunk_id: c.chunk_id.clone(),
            file_path: c.file_path.clone(),
            text: c.text.clone(),
            start: c.start,
            end: c.end,
        }))
    } else {
        Ok(None)
    }
}

/// Get content for a specific symbol
pub fn get_symbol_content(
    request: &SymbolContentRequest,
    symbols: &std::collections::HashMap<String, doctown::docpack::SymbolEntry>,
    chunks: &[doctown::docpack::ChunkEntry],
    archive: &mut ZipArchive<std::fs::File>,
) -> Result<Option<SymbolContentResponse>> {
    if let Some(entry) = symbols.get(&request.symbol) {
        // Determine the file path - use chunks if file field is empty
        let file_path = if entry.file.is_empty() {
            // Try to find the file from chunk data
            if let Some(chunk) = chunks.iter().find(|c| c.chunk_id == entry.chunk) {
                chunk.file_path.clone()
            } else {
                return Ok(None);
            }
        } else {
            entry.file.clone()
        };

        // Read the file containing the symbol
        let file_request = FileRequest {
            path: file_path.clone(),
        };

        match read_file(&file_request, archive) {
            Ok(file_response) => {
                // Parse chunk range to extract the relevant portion
                let content = if entry.chunk.contains('-') {
                    extract_chunk_from_content(&file_response.content, &entry.chunk)
                } else {
                    file_response.content.clone()
                };

                Ok(Some(SymbolContentResponse {
                    symbol: request.symbol.clone(),
                    file: file_path,
                    content,
                    chunk_range: entry.chunk.clone(),
                }))
            }
            Err(e) => {
                eprintln!("Warning: Failed to read file '{}' for symbol '{}': {}", file_path, request.symbol, e);
                Ok(None)
            }
        }
    } else {
        Ok(None)
    }
}

/// Extract a chunk from file content based on byte or line range
fn extract_chunk_from_content(content: &str, chunk_range: &str) -> String {
    let parts: Vec<&str> = chunk_range.split('-').collect();
    if parts.len() != 2 {
        return content.to_string();
    }

    if let (Ok(start), Ok(end)) = (parts[0].parse::<usize>(), parts[1].parse::<usize>()) {
        // Handle byte offsets with proper UTF-8 boundary checking
        let bytes = content.as_bytes();
        if start < bytes.len() && end <= bytes.len() {
            // Find valid UTF-8 boundaries
            let start_pos = std::cmp::min(start, bytes.len());
            let end_pos = std::cmp::min(end, bytes.len());
            
            // Try to extract at exact positions, fallback to full content if invalid UTF-8
            if let Ok(extracted) = std::str::from_utf8(&bytes[start_pos..end_pos]) {
                return extracted.to_string();
            }
        }
    }

    content.to_string()
}

/// List all files in the docpack
pub fn list_files(file_structure: &[doctown::docpack::FileStructureNode]) -> Result<Vec<String>> {
    let mut files = Vec::new();
    collect_files(file_structure, &mut files);
    Ok(files)
}

fn collect_files(nodes: &[doctown::docpack::FileStructureNode], files: &mut Vec<String>) {
    for node in nodes {
        if node.node_type == "file" {
            files.push(node.path.clone());
        }
        if !node.children.is_empty() {
            collect_files(&node.children, files);
        }
    }
}
