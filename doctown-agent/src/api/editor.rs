// editor.rs - Read-only file content access and chunk retrieval

use anyhow::{Result, Context};
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
    // Try to find the file in source/ directory
    let file_path = format!("source/{}", request.path.trim_start_matches('/'));
    
    let mut file = archive.by_name(&file_path)
        .context(format!("File not found: {}", request.path))?;
    
    let mut content = String::new();
    file.read_to_string(&mut content)
        .context("Failed to read file content")?;
    
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
    archive: &mut ZipArchive<std::fs::File>,
) -> Result<Option<SymbolContentResponse>> {
    if let Some(entry) = symbols.get(&request.symbol) {
        // Read the file containing the symbol
        let file_request = FileRequest {
            path: entry.file.clone(),
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
                    file: entry.file.clone(),
                    content,
                    chunk_range: entry.chunk.clone(),
                }))
            }
            Err(_) => Ok(None),
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
        // Assume byte offsets for now
        if start < content.len() && end <= content.len() {
            return content[start..end].to_string();
        }
    }
    
    content.to_string()
}

/// List all files in the docpack
pub fn list_files(
    file_structure: &[doctown::docpack::FileStructureNode],
) -> Result<Vec<String>> {
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
