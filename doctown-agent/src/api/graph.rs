// graph.rs - Graph traversal and relationship queries

use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImpactQuery {
    pub symbol: String,
    pub max_depth: Option<usize>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImpactResponse {
    pub symbol: String,
    pub direct_impacts: Vec<String>,
    pub all_impacts: Vec<String>,
    pub depth: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DependencyQuery {
    pub symbol: String,
    pub max_depth: Option<usize>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DependencyResponse {
    pub symbol: String,
    pub direct_dependencies: Vec<String>,
    pub all_dependencies: Vec<String>,
    pub depth: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PathQuery {
    pub from: String,
    pub to: String,
    pub max_depth: Option<usize>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PathResponse {
    pub from: String,
    pub to: String,
    pub path: Option<Vec<String>>,
    pub length: usize,
}

/// Get all symbols that depend on the given symbol (forward impact)
pub fn get_impact(
    query: &ImpactQuery,
    impact_graph: &HashMap<String, Vec<String>>,
) -> Result<ImpactResponse> {
    let max_depth = query.max_depth.unwrap_or(5);

    let direct = impact_graph.get(&query.symbol).cloned().unwrap_or_default();

    let mut all = HashSet::new();
    let mut visited = HashSet::new();
    let mut queue = vec![(query.symbol.clone(), 0)];

    while let Some((current, depth)) = queue.pop() {
        if depth >= max_depth || visited.contains(&current) {
            continue;
        }
        visited.insert(current.clone());

        if let Some(impacts) = impact_graph.get(&current) {
            for impact in impacts {
                all.insert(impact.clone());
                if depth + 1 < max_depth {
                    queue.push((impact.clone(), depth + 1));
                }
            }
        }
    }

    Ok(ImpactResponse {
        symbol: query.symbol.clone(),
        direct_impacts: direct,
        all_impacts: all.into_iter().collect(),
        depth: max_depth,
    })
}

/// Get all symbols that the given symbol depends on (reverse impact)
pub fn get_dependencies(
    query: &DependencyQuery,
    graph: &doctown::docpack::GraphFile,
) -> Result<DependencyResponse> {
    let max_depth = query.max_depth.unwrap_or(5);

    // Build reverse dependency map from edges
    let mut dep_map: HashMap<String, Vec<String>> = HashMap::new();
    for edge in &graph.edges {
        if edge.edge_type == "calls" || edge.edge_type == "imports" || edge.edge_type == "uses" {
            dep_map
                .entry(edge.from.clone())
                .or_default()
                .push(edge.to.clone());
        }
    }

    let direct = dep_map.get(&query.symbol).cloned().unwrap_or_default();

    let mut all = HashSet::new();
    let mut visited = HashSet::new();
    let mut queue = vec![(query.symbol.clone(), 0)];

    while let Some((current, depth)) = queue.pop() {
        if depth >= max_depth || visited.contains(&current) {
            continue;
        }
        visited.insert(current.clone());

        if let Some(deps) = dep_map.get(&current) {
            for dep in deps {
                all.insert(dep.clone());
                if depth + 1 < max_depth {
                    queue.push((dep.clone(), depth + 1));
                }
            }
        }
    }

    Ok(DependencyResponse {
        symbol: query.symbol.clone(),
        direct_dependencies: direct,
        all_dependencies: all.into_iter().collect(),
        depth: max_depth,
    })
}

/// Find shortest path between two symbols in the graph
pub fn find_path(query: &PathQuery, graph: &doctown::docpack::GraphFile) -> Result<PathResponse> {
    let max_depth = query.max_depth.unwrap_or(10);

    // Build adjacency map
    let mut adj: HashMap<String, Vec<String>> = HashMap::new();
    for edge in &graph.edges {
        adj.entry(edge.from.clone())
            .or_default()
            .push(edge.to.clone());
    }

    // BFS to find shortest path
    let mut queue = vec![(query.from.clone(), vec![query.from.clone()])];
    let mut visited = HashSet::new();
    visited.insert(query.from.clone());

    while let Some((current, path)) = queue.pop() {
        if current == query.to {
            return Ok(PathResponse {
                from: query.from.clone(),
                to: query.to.clone(),
                path: Some(path.clone()),
                length: path.len() - 1,
            });
        }

        if path.len() > max_depth {
            continue;
        }

        if let Some(neighbors) = adj.get(&current) {
            for neighbor in neighbors {
                if !visited.contains(neighbor) {
                    visited.insert(neighbor.clone());
                    let mut new_path = path.clone();
                    new_path.push(neighbor.clone());
                    queue.insert(0, (neighbor.clone(), new_path)); // BFS: insert at front
                }
            }
        }
    }

    Ok(PathResponse {
        from: query.from.clone(),
        to: query.to.clone(),
        path: None,
        length: 0,
    })
}

/// Get related symbols (direct connections)
pub fn get_related_symbols(
    symbol: &str,
    symbols: &HashMap<String, doctown::docpack::SymbolEntry>,
) -> Result<Vec<String>> {
    if let Some(entry) = symbols.get(symbol) {
        Ok(entry.related_symbols.clone())
    } else {
        Ok(vec![])
    }
}
