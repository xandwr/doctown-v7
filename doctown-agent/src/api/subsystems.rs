// subsystems.rs - Query and explore detected subsystems/communities

use anyhow::Result;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SubsystemQuery {
    pub name: Option<String>,
    pub min_confidence: Option<f32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SubsystemResponse {
    pub subsystems: Vec<doctown::docpack::Subsystem>,
    pub total: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SubsystemDetail {
    pub subsystem: doctown::docpack::Subsystem,
    pub symbol_details: Vec<SymbolInfo>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SymbolInfo {
    pub name: String,
    pub kind: String,
    pub file: String,
    pub summary: String,
}

/// List all subsystems matching the query
pub fn list_subsystems(
    query: &SubsystemQuery,
    agent_index: &doctown::docpack::AgentIndex,
) -> Result<SubsystemResponse> {
    let min_conf = query.min_confidence.unwrap_or(0.0);
    
    let mut subsystems: Vec<doctown::docpack::Subsystem> = agent_index.subsystems
        .iter()
        .filter(|s| {
            s.confidence >= min_conf &&
            (query.name.is_none() || s.name.contains(query.name.as_ref().unwrap()))
        })
        .cloned()
        .collect();
    
    let total = subsystems.len();
    
    Ok(SubsystemResponse {
        subsystems,
        total,
    })
}

/// Get detailed information about a specific subsystem
pub fn get_subsystem_detail(
    name: &str,
    agent_index: &doctown::docpack::AgentIndex,
) -> Result<Option<SubsystemDetail>> {
    let subsystem = agent_index.subsystems
        .iter()
        .find(|s| s.name == name)
        .cloned();
    
    if let Some(sub) = subsystem {
        let mut symbol_details = Vec::new();
        
        for symbol_name in &sub.symbols {
            if let Some(entry) = agent_index.symbols.get(symbol_name) {
                symbol_details.push(SymbolInfo {
                    name: symbol_name.clone(),
                    kind: entry.kind.clone(),
                    file: entry.file.clone(),
                    summary: entry.summary.clone(),
                });
            }
        }
        
        Ok(Some(SubsystemDetail {
            subsystem: sub,
            symbol_details,
        }))
    } else {
        Ok(None)
    }
}

/// Get all subsystems that contain a specific file
pub fn get_subsystems_by_file(
    file_path: &str,
    agent_index: &doctown::docpack::AgentIndex,
) -> Result<Vec<doctown::docpack::Subsystem>> {
    let subsystems = agent_index.subsystems
        .iter()
        .filter(|s| s.files.contains(&file_path.to_string()))
        .cloned()
        .collect();
    
    Ok(subsystems)
}

/// Get all subsystems that contain a specific symbol
pub fn get_subsystem_by_symbol(
    symbol: &str,
    agent_index: &doctown::docpack::AgentIndex,
) -> Result<Option<String>> {
    if let Some(entry) = agent_index.symbols.get(symbol) {
        Ok(Some(entry.subsystem.clone()))
    } else {
        Ok(None)
    }
}
