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
    results.sort_by(|a, b| {
        b.score
            .partial_cmp(&a.score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

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
        symbols
            .keys()
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

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SignatureSearchQuery {
    pub query: String,
    pub limit: Option<usize>,
    pub filter_subsystem: Option<String>,
    pub filter_kind: Option<String>,
}

/// Search for symbols by signature pattern matching
/// Supports partial matches like "fn new() -> Result", "impl Iterator for", etc.
pub fn signature_search(
    query: &SignatureSearchQuery,
    symbols: &HashMap<String, doctown::docpack::SymbolEntry>,
) -> Result<SearchResponse> {
    let query_lower = query.query.to_lowercase();
    let limit = query.limit.unwrap_or(20);

    let mut results: Vec<SearchResult> = Vec::new();

    for (name, entry) in symbols.iter() {
        // Skip symbols without signatures
        let signature = match &entry.signature {
            Some(sig) => sig,
            None => continue,
        };

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

        let signature_lower = signature.to_lowercase();

        // Scoring based on signature matching
        let mut score = 0.0;

        // Exact signature match
        if signature_lower == query_lower {
            score += 100.0;
        }
        // Full substring match in signature
        else if signature_lower.contains(&query_lower) {
            score += 50.0;
        }
        // Token-based fuzzy matching: split query and signature into tokens
        else {
            // Extract meaningful tokens from the query (skip common keywords for better matching)
            let query_tokens: Vec<&str> = query_lower
                .split(|c: char| !c.is_alphanumeric() && c != '_')
                .filter(|s| !s.is_empty())
                .filter(|s| !matches!(*s, "fn" | "impl" | "for" | "pub" | "async" | "const" | "unsafe" | "extern"))
                .collect();

            // Extract meaningful tokens from the signature
            let signature_tokens: Vec<&str> = signature_lower
                .split(|c: char| !c.is_alphanumeric() && c != '_')
                .filter(|s| !s.is_empty())
                .collect();

            let mut matched_tokens = 0;
            for query_token in &query_tokens {
                if signature_tokens.iter().any(|sig_token| sig_token.contains(query_token)) {
                    matched_tokens += 1;
                }
            }

            if !query_tokens.is_empty() {
                let match_ratio = matched_tokens as f32 / query_tokens.len() as f32;
                if match_ratio > 0.0 {
                    score += match_ratio * 30.0;
                }
            }
        }

        // Bonus for exact return type matches (looking for "-> Type" patterns)
        if query_lower.contains("->") && signature_lower.contains("->") {
            // Extract return type from query
            if let Some(query_ret) = query_lower.split("->").nth(1) {
                let query_ret = query_ret.trim();
                if let Some(sig_ret) = signature_lower.split("->").nth(1) {
                    let sig_ret = sig_ret.trim();
                    if sig_ret.contains(query_ret) {
                        score += 20.0;
                    }
                }
            }
        }

        // Bonus for trait impl patterns (e.g., "impl Iterator for")
        if query_lower.contains("impl") && signature_lower.contains("impl") {
            score += 15.0;
        }

        if score > 0.0 {
            results.push(SearchResult {
                symbol: name.clone(),
                kind: entry.kind.clone(),
                file: entry.file.clone(),
                subsystem: entry.subsystem.clone(),
                summary: format!("{}\nSignature: {}", entry.summary, signature),
                score,
            });
        }
    }

    // Sort by score descending
    results.sort_by(|a, b| {
        b.score
            .partial_cmp(&a.score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let total = results.len();
    results.truncate(limit);

    Ok(SearchResponse {
        results,
        total,
        query: query.query.clone(),
    })
}
