// tasks.rs - Task-oriented views of the codebase

use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TaskQuery {
    pub task_type: String, // e.g., "add_feature", "fix_bug", "refactor"
    pub context: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TaskResponse {
    pub task_type: String,
    pub relevant_symbols: Vec<String>,
    pub suggested_entry_points: Vec<String>,
    pub related_files: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuickstartResponse {
    pub entry_points: Vec<String>,
    pub core_types: Vec<String>,
    pub most_connected: Vec<String>,
    pub subsystem_map: HashMap<String, String>,
}

/// Get the quickstart guide for navigating the codebase
pub fn get_quickstart(
    agent_index: &doctown::docpack::AgentIndex,
) -> Result<QuickstartResponse> {
    Ok(QuickstartResponse {
        entry_points: agent_index.quickstart.entry_points.clone(),
        core_types: agent_index.quickstart.core_types.clone(),
        most_connected: agent_index.quickstart.most_connected.clone(),
        subsystem_map: agent_index.quickstart.subsystem_map.clone(),
    })
}

/// Get symbols relevant to a specific task
pub fn get_task_view(
    query: &TaskQuery,
    agent_index: &doctown::docpack::AgentIndex,
) -> Result<TaskResponse> {
    // Check if we have a pre-computed task view
    if let Some(symbols) = agent_index.tasks.get(&query.task_type) {
        return Ok(TaskResponse {
            task_type: query.task_type.clone(),
            relevant_symbols: symbols.clone(),
            suggested_entry_points: agent_index.quickstart.entry_points.clone(),
            related_files: vec![],
        });
    }
    
    // Otherwise, try to construct a task view based on context
    let relevant = if let Some(context) = &query.context {
        // Simple keyword matching for now
        let context_lower = context.to_lowercase();
        agent_index.symbols.keys()
            .filter(|s| {
                let s_lower = s.to_lowercase();
                s_lower.contains(&context_lower) || 
                agent_index.symbols.get(*s)
                    .map(|e| e.summary.to_lowercase().contains(&context_lower))
                    .unwrap_or(false)
            })
            .cloned()
            .collect()
    } else {
        vec![]
    };
    
    Ok(TaskResponse {
        task_type: query.task_type.clone(),
        relevant_symbols: relevant,
        suggested_entry_points: agent_index.quickstart.entry_points.clone(),
        related_files: vec![],
    })
}

/// List all available task types in the index
pub fn list_task_types(
    agent_index: &doctown::docpack::AgentIndex,
) -> Result<Vec<String>> {
    Ok(agent_index.tasks.keys().cloned().collect())
}

/// Get entry points for the codebase
pub fn get_entry_points(
    agent_index: &doctown::docpack::AgentIndex,
) -> Result<Vec<String>> {
    Ok(agent_index.quickstart.entry_points.clone())
}

/// Get core types in the codebase
pub fn get_core_types(
    agent_index: &doctown::docpack::AgentIndex,
) -> Result<Vec<String>> {
    Ok(agent_index.quickstart.core_types.clone())
}

/// Get most connected symbols (hub nodes)
pub fn get_most_connected(
    agent_index: &doctown::docpack::AgentIndex,
) -> Result<Vec<String>> {
    Ok(agent_index.quickstart.most_connected.clone())
}
