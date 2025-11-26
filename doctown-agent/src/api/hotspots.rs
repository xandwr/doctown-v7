// hotspots.rs - Hot spots analysis for identifying critical code areas
//
// This module analyzes the codebase to identify "hot spots" - areas that are:
// - Most modified (from git history)
// - Most connected (from dependency graph)
// - High complexity (cyclomatic complexity bombs)
// - Orphaned (low connectivity, potentially dead code)

use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::path::Path;
use std::process::Command;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HotSpotsQuery {
    /// Maximum number of items to return in each category
    pub limit: Option<usize>,
    /// Path to the git repository (defaults to current directory)
    pub repo_path: Option<String>,
    /// Number of commits to analyze for modification history
    pub commit_depth: Option<usize>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HotSpotsResponse {
    /// Files/symbols most frequently modified (from git history)
    pub most_modified: Vec<ModificationHotSpot>,
    /// Symbols with highest connectivity (from dependency graph)
    pub most_connected: Vec<ConnectivityHotSpot>,
    /// Functions/symbols with high cyclomatic complexity
    pub complexity_bombs: Vec<ComplexityHotSpot>,
    /// Symbols with low connectivity (potential dead code)
    pub orphans: Vec<OrphanHotSpot>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModificationHotSpot {
    pub file_path: String,
    pub modification_count: usize,
    pub lines_changed: usize,
    pub last_modified: String,
    pub contributors: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConnectivityHotSpot {
    pub symbol: String,
    pub incoming_connections: usize,
    pub outgoing_connections: usize,
    pub total_connections: usize,
    pub file_path: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComplexityHotSpot {
    pub symbol: String,
    pub file_path: String,
    pub complexity_score: f32,
    pub complexity_estimate: String,
    pub symbol_kind: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OrphanHotSpot {
    pub symbol: String,
    pub file_path: String,
    pub connection_count: usize,
    pub symbol_kind: String,
    pub is_exported: bool,
}

/// Analyze hot spots in the codebase
pub fn analyze_hotspots(
    query: &HotSpotsQuery,
    agent_index: &doctown::docpack::AgentIndex,
    graph: &doctown::docpack::GraphFile,
    documentation: &doctown::docpack::DocumentationFile,
) -> Result<HotSpotsResponse> {
    let limit = query.limit.unwrap_or(10);

    // Analyze most modified files from git history
    let most_modified = analyze_most_modified(query, limit)?;

    // Analyze most connected symbols from the graph
    let most_connected = analyze_most_connected(agent_index, graph, limit)?;

    // Identify complexity bombs
    let complexity_bombs = analyze_complexity_bombs(documentation, limit)?;

    // Identify orphaned code
    let orphans = analyze_orphans(agent_index, graph, limit)?;

    Ok(HotSpotsResponse {
        most_modified,
        most_connected,
        complexity_bombs,
        orphans,
    })
}

/// Analyze files by modification frequency from git history
fn analyze_most_modified(query: &HotSpotsQuery, limit: usize) -> Result<Vec<ModificationHotSpot>> {
    let repo_path = query.repo_path.as_deref().unwrap_or(".");
    let commit_depth = query.commit_depth.unwrap_or(100);

    // Check if we're in a git repository
    if !Path::new(repo_path).join(".git").exists() {
        return Ok(vec![]);
    }

    // Get git log with file statistics
    let output = Command::new("git")
        .arg("-C")
        .arg(repo_path)
        .arg("log")
        .arg(format!("-{}", commit_depth))
        .arg("--name-only")
        .arg("--format=%H|%an|%ad")
        .arg("--date=short")
        .output()?;

    if !output.status.success() {
        return Ok(vec![]);
    }

    let log_output = String::from_utf8_lossy(&output.stdout);

    // Parse git log to count modifications per file
    let mut file_modifications: HashMap<String, FileModificationData> = HashMap::new();
    let mut current_commit_info: Option<(String, String)> = None; // (author, date)

    for line in log_output.lines() {
        let line = line.trim();

        if line.is_empty() {
            continue;
        }

        // Parse commit header: hash|author|date
        if line.contains('|') {
            let parts: Vec<&str> = line.split('|').collect();
            if parts.len() >= 3 {
                current_commit_info = Some((parts[1].to_string(), parts[2].to_string()));
            }
            continue;
        }

        // This is a file path
        if let Some((author, date)) = &current_commit_info {
            let entry = file_modifications.entry(line.to_string()).or_insert_with(|| {
                FileModificationData {
                    modification_count: 0,
                    last_modified: date.clone(),
                    contributors: HashSet::new(),
                }
            });

            entry.modification_count += 1;
            entry.contributors.insert(author.clone());

            // Update last modified date (more recent)
            if date > &entry.last_modified {
                entry.last_modified = date.clone();
            }
        }
    }

    // Get line change statistics for each file
    let mut hotspots: Vec<ModificationHotSpot> = vec![];

    for (file_path, data) in file_modifications.iter() {
        // Get total lines changed for this file
        let lines_changed = get_file_line_changes(repo_path, file_path, commit_depth)?;

        hotspots.push(ModificationHotSpot {
            file_path: file_path.clone(),
            modification_count: data.modification_count,
            lines_changed,
            last_modified: data.last_modified.clone(),
            contributors: data.contributors.iter().cloned().collect(),
        });
    }

    // Sort by modification count
    hotspots.sort_by(|a, b| b.modification_count.cmp(&a.modification_count));
    hotspots.truncate(limit);

    Ok(hotspots)
}

struct FileModificationData {
    modification_count: usize,
    last_modified: String,
    contributors: HashSet<String>,
}

/// Get total line changes for a file from git history
fn get_file_line_changes(repo_path: &str, file_path: &str, commit_depth: usize) -> Result<usize> {
    let output = Command::new("git")
        .arg("-C")
        .arg(repo_path)
        .arg("log")
        .arg(format!("-{}", commit_depth))
        .arg("--numstat")
        .arg("--format=")
        .arg("--")
        .arg(file_path)
        .output()?;

    if !output.status.success() {
        return Ok(0);
    }

    let log_output = String::from_utf8_lossy(&output.stdout);
    let mut total_changes = 0;

    for line in log_output.lines() {
        let parts: Vec<&str> = line.split_whitespace().collect();
        if parts.len() >= 2 {
            if let (Ok(added), Ok(deleted)) = (parts[0].parse::<usize>(), parts[1].parse::<usize>()) {
                total_changes += added + deleted;
            }
        }
    }

    Ok(total_changes)
}

/// Analyze symbols by connectivity in the dependency graph
fn analyze_most_connected(
    agent_index: &doctown::docpack::AgentIndex,
    graph: &doctown::docpack::GraphFile,
    limit: usize,
) -> Result<Vec<ConnectivityHotSpot>> {
    // Count incoming and outgoing connections for each symbol
    let mut connectivity_map: HashMap<String, (usize, usize)> = HashMap::new();

    for edge in &graph.edges {
        // Count outgoing edges from source
        let outgoing = connectivity_map.entry(edge.from.clone()).or_insert((0, 0));
        outgoing.1 += 1;

        // Count incoming edges to target
        let incoming = connectivity_map.entry(edge.to.clone()).or_insert((0, 0));
        incoming.0 += 1;
    }

    // Build hotspots from symbols with high connectivity
    let mut hotspots: Vec<ConnectivityHotSpot> = vec![];

    for (symbol_name, (incoming, outgoing)) in connectivity_map.iter() {
        // Try to find the symbol in the agent index
        let file_path = agent_index.symbols
            .get(symbol_name)
            .map(|s| s.file.clone())
            .unwrap_or_else(|| "unknown".to_string());

        hotspots.push(ConnectivityHotSpot {
            symbol: symbol_name.clone(),
            incoming_connections: *incoming,
            outgoing_connections: *outgoing,
            total_connections: incoming + outgoing,
            file_path,
        });
    }

    // Sort by total connections
    hotspots.sort_by(|a, b| b.total_connections.cmp(&a.total_connections));
    hotspots.truncate(limit);

    Ok(hotspots)
}

/// Identify functions/symbols with high complexity
fn analyze_complexity_bombs(
    documentation: &doctown::docpack::DocumentationFile,
    limit: usize,
) -> Result<Vec<ComplexityHotSpot>> {
    let mut hotspots: Vec<ComplexityHotSpot> = vec![];

    // Check files for high complexity
    for file_summary in &documentation.file_summaries {
        if file_summary.complexity_score > 3.0 {
            hotspots.push(ComplexityHotSpot {
                symbol: format!("file:{}", file_summary.file_path),
                file_path: file_summary.file_path.clone(),
                complexity_score: file_summary.complexity_score,
                complexity_estimate: if file_summary.complexity_score > 5.0 {
                    "High".to_string()
                } else if file_summary.complexity_score > 3.0 {
                    "Medium".to_string()
                } else {
                    "Low".to_string()
                },
                symbol_kind: "file".to_string(),
            });
        }
    }

    // Check functions for high complexity
    for function_doc in &documentation.function_docs {
        if function_doc.complexity_estimate == "High" {
            // Estimate a numeric score based on the complexity estimate
            let complexity_score = match function_doc.complexity_estimate.as_str() {
                "High" => 4.5,
                "Medium" => 2.5,
                _ => 1.0,
            };

            hotspots.push(ComplexityHotSpot {
                symbol: function_doc.name.clone(),
                file_path: function_doc.file_path.clone(),
                complexity_score,
                complexity_estimate: function_doc.complexity_estimate.clone(),
                symbol_kind: "function".to_string(),
            });
        }
    }

    // Sort by complexity score
    hotspots.sort_by(|a, b| {
        b.complexity_score.partial_cmp(&a.complexity_score).unwrap_or(std::cmp::Ordering::Equal)
    });
    hotspots.truncate(limit);

    Ok(hotspots)
}

/// Identify orphaned symbols (low connectivity, potential dead code)
fn analyze_orphans(
    agent_index: &doctown::docpack::AgentIndex,
    graph: &doctown::docpack::GraphFile,
    limit: usize,
) -> Result<Vec<OrphanHotSpot>> {
    // Count connections for each symbol
    let mut connection_count: HashMap<String, usize> = HashMap::new();

    for edge in &graph.edges {
        *connection_count.entry(edge.from.clone()).or_insert(0) += 1;
        *connection_count.entry(edge.to.clone()).or_insert(0) += 1;
    }

    let mut orphans: Vec<OrphanHotSpot> = vec![];

    for (symbol_name, symbol_entry) in agent_index.symbols.iter() {
        let connections = connection_count.get(symbol_name).copied().unwrap_or(0);

        // Consider it an orphan if it has 2 or fewer connections
        if connections <= 2 {
            // Skip common patterns that are expected to be low-connectivity
            if should_skip_orphan_check(symbol_name, symbol_entry) {
                continue;
            }

            // Check if symbol appears to be exported (heuristic based on naming and kind)
            let is_exported = is_likely_exported(symbol_entry);

            orphans.push(OrphanHotSpot {
                symbol: symbol_name.clone(),
                file_path: symbol_entry.file.clone(),
                connection_count: connections,
                symbol_kind: symbol_entry.kind.clone(),
                is_exported,
            });
        }
    }

    // Sort by connection count (ascending - least connected first)
    orphans.sort_by(|a, b| a.connection_count.cmp(&b.connection_count));
    orphans.truncate(limit);

    Ok(orphans)
}

/// Check if a symbol should be skipped from orphan detection
fn should_skip_orphan_check(symbol_name: &str, symbol: &doctown::docpack::SymbolEntry) -> bool {
    // Skip test functions
    if symbol_name.starts_with("test_") || symbol_name.contains("_test") {
        return true;
    }

    // Skip main functions
    if symbol_name == "main" {
        return true;
    }

    // Skip constants and statics (they're often intentionally standalone)
    if symbol.kind == "const" || symbol.kind == "static" {
        return true;
    }

    // Skip macro definitions
    if symbol.kind == "macro" {
        return true;
    }

    false
}

/// Heuristic to determine if a symbol is likely exported
fn is_likely_exported(symbol: &doctown::docpack::SymbolEntry) -> bool {
    // Check the signature for pub keyword if available
    if let Some(sig) = &symbol.signature {
        if sig.contains("pub ") {
            return true;
        }
    }

    // Structs, enums, and traits at module level are often exported
    if symbol.kind == "struct" || symbol.kind == "enum" || symbol.kind == "trait" {
        return true;
    }

    false
}
