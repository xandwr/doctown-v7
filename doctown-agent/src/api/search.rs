// search.rs - Semantic and keyword search over the docpack index

use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchQuery {
    pub query: String,
    pub limit: Option<usize>,
    pub filter_subsystem: Option<String>,
    pub filter_kind: Option<String>, // "function", "struct", "type", etc.
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchResult {
    pub symbol: String,
    pub kind: String,
    pub file: String,
    pub subsystem: String,
    pub summary: String,
    pub score: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchResponse {
    pub results: Vec<SearchResult>,
    pub total: usize,
    pub query: String,
}

/// Perform semantic search over symbols using embeddings
pub fn semantic_search(
    query: &SearchQuery,
    symbols: &HashMap<String, doctown::docpack::SymbolEntry>,
    embeddings: &[doctown::ingest::Embedding],
) -> Result<SearchResponse> {
    // TODO: Implement actual embedding-based search
    // For now, do simple keyword matching
    let results = keyword_search(query, symbols)?;
    Ok(results)
}

/// Perform keyword-based search over symbol names and summaries
pub fn keyword_search(
    query: &SearchQuery,
    symbols: &HashMap<String, doctown::docpack::SymbolEntry>,
) -> Result<SearchResponse> {
    let query_lower = query.query.to_lowercase();
    let limit = query.limit.unwrap_or(20);
    
    let mut results: Vec<SearchResult> = Vec::new();
    
    for (name, entry) in symbols.iter() {
        // Apply filters
        if let Some(ref subsystem_filter) = query.filter_subsystem {
            if &entry.subsystem != subsystem_filter {
                continue;
            }
        }
        
        if let Some(ref kind_filter) = query.filter_kind {
            if &entry.kind != kind_filter {
                continue;
            }
        }
        
        // Simple scoring: match in name (higher) or summary (lower)
        let name_lower = name.to_lowercase();
        let summary_lower = entry.summary.to_lowercase();
        
        let mut score = 0.0;
        if name_lower.contains(&query_lower) {
            score += 10.0;
        }
        if summary_lower.contains(&query_lower) {
            score += 5.0;
        }
        
        // Check for exact matches
        if name_lower == query_lower {
            score += 20.0;
        }
        
        if score > 0.0 {
            results.push(SearchResult {
                symbol: name.clone(),
                kind: entry.kind.clone(),
                file: entry.file.clone(),
                subsystem: entry.subsystem.clone(),
                summary: entry.summary.clone(),
                score,
            });
        }
    }
    
    // Sort by score descending
    results.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
    
    let total = results.len();
    results.truncate(limit);
    
    Ok(SearchResponse {
        results,
        total,
        query: query.query.clone(),
    })
}

/// Find symbols by exact name
pub fn get_symbol(
    name: &str,
    symbols: &HashMap<String, doctown::docpack::SymbolEntry>,
) -> Result<Option<doctown::docpack::SymbolEntry>> {
    Ok(symbols.get(name).cloned())
}

/// List all symbols matching a pattern
pub fn list_symbols(
    pattern: Option<&str>,
    symbols: &HashMap<String, doctown::docpack::SymbolEntry>,
    limit: Option<usize>,
) -> Result<Vec<String>> {
    let limit = limit.unwrap_or(100);
    
    let mut names: Vec<String> = if let Some(pat) = pattern {
        let pat_lower = pat.to_lowercase();
        symbols.keys()
            .filter(|k| k.to_lowercase().contains(&pat_lower))
            .cloned()
            .collect()
    } else {
        symbols.keys().cloned().collect()
    };
    
    names.sort();
    names.truncate(limit);
    
    Ok(names)
}
