// agent.rs - Agent-optimized index generation for LLM navigation
//
// Transforms community detection results, symbol tables, and graph structure
// into a curated index designed for AI agents to navigate codebases efficiently.

use crate::docpack::{AgentIndex, AgentQuickstart, Subsystem, SymbolEntry};
use crate::ingest::{GraphEdge, GraphEdgeKind, ProjectGraph, SymbolNode};
use crate::llm::LlmEngine;
use std::collections::{HashMap, HashSet};

/// Build the complete agent index from a project graph
///
/// If `llm_engine` is provided, plain-English summaries will be generated for all symbols
pub fn build_agent_index(graph: &ProjectGraph) -> AgentIndex {
    build_agent_index_with_llm(graph, None)
}

/// Build the complete agent index with optional LLM summary generation
pub fn build_agent_index_with_llm(
    graph: &ProjectGraph,
    llm_engine: Option<&mut LlmEngine>,
) -> AgentIndex {
    let subsystems = build_subsystems(graph);
    let mut symbols = build_symbol_table(graph, &subsystems);

    // Generate LLM summaries if engine is provided
    if let Some(engine) = llm_engine {
        println!("🤖 Generating LLM plain-English summaries...");
        let symbol_count = symbols.len();
        let mut processed = 0;

        for (symbol_name, entry) in symbols.iter_mut() {
            processed += 1;
            if processed % 10 == 0 {
                println!("   Progress: {}/{}", processed, symbol_count);
            }

            // Get code context from chunks if available
            let code_context = get_symbol_code_context(graph, symbol_name);

            match engine.explain_symbol(
                symbol_name,
                &entry.kind,
                entry.signature.as_deref(),
                code_context.as_deref(),
                &entry.summary,
            ) {
                Ok(llm_summary) => {
                    entry.llm_summary = Some(llm_summary);
                }
                Err(e) => {
                    eprintln!("⚠️  Failed to generate LLM summary for '{}': {}", symbol_name, e);
                }
            }
        }

        println!("✅ Generated {} LLM summaries", processed);
    }

    let tasks = generate_task_views(graph, &symbols);
    let impact_graph = build_impact_graph(graph);
    let quickstart = generate_quickstart(graph, &subsystems, &symbols, &impact_graph);

    AgentIndex {
        version: "1.0.0".to_string(),
        subsystems,
        symbols,
        tasks,
        impact_graph,
        quickstart,
    }
}

/// Extract code context for a symbol from the project graph
fn get_symbol_code_context(graph: &ProjectGraph, symbol_name: &str) -> Option<String> {
    // Find the symbol in the graph
    let symbol = graph.symbols.iter().find(|s| s.name == symbol_name)?;

    // Get the first chunk for this symbol
    let chunk_id = symbol.chunk_ids.first()?;

    // Find the chunk in the graph
    graph.chunks
        .iter()
        .find(|c| &c.id.0 == chunk_id)
        .map(|c| c.text.clone())
}

/// Convert detected communities into labeled subsystems with role inference
fn build_subsystems(graph: &ProjectGraph) -> Vec<Subsystem> {
    let mut subsystems = Vec::new();

    for community in &graph.communities {
        let confidence = community.cohesion;
        let role = infer_subsystem_role(&community.symbol_names, &community.suggested_label);
        let summary =
            generate_subsystem_summary(&community.symbol_names, &community.suggested_label, &role);

        subsystems.push(Subsystem {
            name: community
                .suggested_label
                .clone()
                .unwrap_or_else(|| format!("subsystem_{}", community.id)),
            symbols: community.symbol_names.clone(),
            files: community.files.iter().cloned().collect(),
            confidence,
            role,
            summary,
        });
    }

    // Sort by confidence (highest first)
    subsystems.sort_by(|a, b| {
        b.confidence
            .partial_cmp(&a.confidence)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    subsystems
}

/// Infer the role/purpose of a subsystem based on its symbols and label
fn infer_subsystem_role(symbol_names: &[String], label: &Option<String>) -> String {
    // Check label first
    if let Some(lbl) = label {
        let lower = lbl.to_lowercase();
        if lower.contains("server") || lower.contains("rpc") || lower.contains("handler") {
            return "RPC/server layer".to_string();
        }
        if lower.contains("cli") || lower.contains("command") || lower.contains("arg") {
            return "CLI interface".to_string();
        }
        if lower.contains("parse") || lower.contains("parser") || lower.contains("syntax") {
            return "parsing and analysis".to_string();
        }
        if lower.contains("embed") || lower.contains("model") || lower.contains("inference") {
            return "embedding and ML inference".to_string();
        }
        if lower.contains("docpack") || lower.contains("archive") || lower.contains("zip") {
            return "archive I/O and packaging".to_string();
        }
        if lower.contains("graph") || lower.contains("edge") || lower.contains("node") {
            return "graph structures and algorithms".to_string();
        }
        if lower.contains("doc") || lower.contains("documentation") || lower.contains("generate") {
            return "documentation generation".to_string();
        }
        if lower.contains("ingest") || lower.contains("process") || lower.contains("extract") {
            return "source code ingestion".to_string();
        }
    }

    // Pattern matching on symbol names
    let symbol_lower: Vec<String> = symbol_names.iter().map(|s| s.to_lowercase()).collect();

    if symbol_lower
        .iter()
        .any(|s| s.contains("server") || s.contains("handler") || s.contains("request"))
    {
        return "RPC/server layer".to_string();
    }
    if symbol_lower
        .iter()
        .any(|s| s.contains("cli") || s.contains("command") || s.contains("args"))
    {
        return "CLI interface".to_string();
    }
    if symbol_lower.iter().any(|s| {
        s.contains("read") || s.contains("write") || s.contains("file") || s.contains("path")
    }) {
        return "I/O operations".to_string();
    }
    if symbol_lower
        .iter()
        .any(|s| s.contains("parse") || s.contains("token") || s.contains("syntax"))
    {
        return "parsing and analysis".to_string();
    }
    if symbol_lower
        .iter()
        .any(|s| s.contains("embed") || s.contains("model") || s.contains("infer"))
    {
        return "embedding and ML inference".to_string();
    }
    if symbol_lower
        .iter()
        .any(|s| s.contains("graph") || s.contains("edge") || s.contains("node"))
    {
        return "graph structures and algorithms".to_string();
    }

    "core logic".to_string()
}

/// Generate a concise summary of what the subsystem does
fn generate_subsystem_summary(
    symbol_names: &[String],
    label: &Option<String>,
    role: &str,
) -> String {
    let label_str = label.as_ref().map(|s| s.as_str()).unwrap_or("unnamed");
    let symbol_count = symbol_names.len();

    format!(
        "Subsystem '{}' ({} symbols): {}",
        label_str, symbol_count, role
    )
}

/// Build cross-references: scan chunks to find test and documentation mentions of symbols
fn build_cross_references(
    graph: &ProjectGraph,
) -> (HashMap<String, Vec<String>>, HashMap<String, Vec<String>>) {
    let mut test_refs: HashMap<String, Vec<String>> = HashMap::new();
    let mut doc_refs: HashMap<String, Vec<String>> = HashMap::new();

    // Build a set of all symbol names for quick lookup
    let symbol_names: HashSet<String> = graph.symbols.iter().map(|s| s.name.clone()).collect();

    // Scan through all chunks to find mentions
    for (idx, chunk) in graph.chunks.iter().enumerate() {
        if let Some(file_path) = graph.chunk_to_file.get(&idx) {
            let is_test = file_path.contains("test")
                || file_path.contains("tests/")
                || file_path.contains("_test.rs")
                || file_path.ends_with("tests.rs");

            let is_doc = file_path.ends_with(".md")
                || file_path.ends_with(".txt")
                || file_path.contains("README")
                || file_path.contains("doc/")
                || file_path.contains("docs/");

            if !is_test && !is_doc {
                continue; // Only interested in test and doc files
            }

            let text = &chunk.text;

            // Find mentions of each symbol in this chunk
            for symbol_name in &symbol_names {
                // Simple word boundary check - look for the symbol name as a word
                if text.contains(symbol_name) {
                    // Verify it's a word boundary (not part of another identifier)
                    let pattern = format!(r"\b{}\b", regex::escape(symbol_name));
                    if regex::Regex::new(&pattern).ok().map_or(false, |re| re.is_match(text)) {
                        let location = format!("{}:{}", file_path, chunk.id.0);

                        if is_test {
                            test_refs.entry(symbol_name.clone())
                                .or_insert_with(Vec::new)
                                .push(location.clone());
                        }
                        if is_doc {
                            doc_refs.entry(symbol_name.clone())
                                .or_insert_with(Vec::new)
                                .push(location.clone());
                        }
                    }
                }
            }
        }
    }

    // Deduplicate and sort
    for refs in test_refs.values_mut() {
        refs.sort();
        refs.dedup();
    }
    for refs in doc_refs.values_mut() {
        refs.sort();
        refs.dedup();
    }

    (test_refs, doc_refs)
}

/// Extract concrete usage examples for each symbol from test files and documentation
fn extract_usage_examples(
    graph: &ProjectGraph,
) -> HashMap<String, Vec<crate::docpack::UsageExample>> {
    use crate::docpack::UsageExample;
    let mut examples: HashMap<String, Vec<UsageExample>> = HashMap::new();

    // Build a set of all symbol names for quick lookup
    let symbol_names: HashSet<String> = graph.symbols.iter().map(|s| s.name.clone()).collect();

    // Scan through all chunks to find usage examples
    for (idx, chunk) in graph.chunks.iter().enumerate() {
        if let Some(file_path) = graph.chunk_to_file.get(&idx) {
            // Determine the type of example
            let is_test = file_path.contains("test")
                || file_path.contains("tests/")
                || file_path.contains("_test.rs")
                || file_path.ends_with("tests.rs");

            let is_doc = file_path.ends_with(".md")
                || file_path.ends_with(".txt")
                || file_path.contains("README")
                || file_path.contains("doc/")
                || file_path.contains("docs/");

            if !is_test && !is_doc {
                continue; // Only extract examples from test and doc files for now
            }

            let example_type = if is_test { "test" } else { "doc" };
            let text = &chunk.text;

            // Find mentions of each symbol in this chunk
            for symbol_name in &symbol_names {
                if text.contains(symbol_name) {
                    // Verify it's a word boundary (not part of another identifier)
                    let pattern = format!(r"\b{}\b", regex::escape(symbol_name));
                    if regex::Regex::new(&pattern).ok().map_or(false, |re| re.is_match(text)) {
                        // Extract a context snippet around the symbol usage
                        let context = extract_context_snippet(text, symbol_name, 150);

                        // Parse line numbers from chunk ID if available
                        let (line_start, line_end) = parse_chunk_lines(&chunk.id.0);

                        let example = UsageExample {
                            file: file_path.clone(),
                            context,
                            example_type: example_type.to_string(),
                            line_start,
                            line_end,
                        };

                        examples.entry(symbol_name.clone())
                            .or_insert_with(Vec::new)
                            .push(example);
                    }
                }
            }
        }
    }

    // Limit to top 5 examples per symbol and deduplicate
    for examples_list in examples.values_mut() {
        examples_list.sort_by(|a, b| {
            // Prioritize tests over docs
            match (a.example_type.as_str(), b.example_type.as_str()) {
                ("test", "doc") => std::cmp::Ordering::Less,
                ("doc", "test") => std::cmp::Ordering::Greater,
                _ => a.file.cmp(&b.file),
            }
        });
        examples_list.truncate(5);
    }

    examples
}

/// Extract a context snippet around a symbol usage
fn extract_context_snippet(text: &str, symbol_name: &str, max_chars: usize) -> String {
    // Find the position of the symbol
    if let Some(pos) = text.find(symbol_name) {
        // Extract surrounding context
        let start = pos.saturating_sub(max_chars / 2);
        let end = (pos + symbol_name.len() + max_chars / 2).min(text.len());

        let mut snippet = text[start..end].to_string();

        // Trim to complete lines for better readability
        if let Some(first_newline) = snippet.find('\n') {
            if start > 0 {
                snippet = snippet[first_newline + 1..].to_string();
            }
        }
        if let Some(last_newline) = snippet.rfind('\n') {
            if end < text.len() {
                snippet = snippet[..last_newline].to_string();
            }
        }

        snippet.trim().to_string()
    } else {
        text.chars().take(max_chars).collect()
    }
}

/// Parse line numbers from chunk ID (format: "start_line-end_line" or "byte:start-end")
fn parse_chunk_lines(chunk_id: &str) -> (usize, usize) {
    // Try to parse line-based chunk ID first
    if let Some((start, end)) = chunk_id.split_once('-') {
        if let (Ok(s), Ok(e)) = (start.parse::<usize>(), end.parse::<usize>()) {
            return (s, e);
        }
    }

    // Default to 0 if we can't parse
    (0, 0)
}

/// Build comprehensive symbol table with all metadata agents need
fn build_symbol_table(
    graph: &ProjectGraph,
    subsystems: &[Subsystem],
) -> HashMap<String, SymbolEntry> {
    let mut symbols = HashMap::new();

    // Build symbol name -> subsystem mapping
    let mut symbol_to_subsystem: HashMap<String, String> = HashMap::new();
    for subsystem in subsystems {
        for symbol_name in &subsystem.symbols {
            symbol_to_subsystem.insert(symbol_name.clone(), subsystem.name.clone());
        }
    }

    // Build symbol name -> related symbols mapping from graph edges
    let mut related_map: HashMap<String, HashSet<String>> = HashMap::new();
    for edge in &graph.edges {
        match edge.kind {
            GraphEdgeKind::SymbolToSymbol | GraphEdgeKind::SymbolSimilarity => {
                related_map
                    .entry(edge.src.clone())
                    .or_insert_with(HashSet::new)
                    .insert(edge.dst.clone());
                related_map
                    .entry(edge.dst.clone())
                    .or_insert_with(HashSet::new)
                    .insert(edge.src.clone());
            }
            _ => {}
        }
    }

    // Build chunk_id -> file mapping
    let mut chunk_to_file: HashMap<String, String> = HashMap::new();
    for (idx, chunk) in graph.chunks.iter().enumerate() {
        if let Some(file_path) = graph.chunk_to_file.get(&idx) {
            chunk_to_file.insert(chunk.id.0.clone(), file_path.clone());
        }
    }

    // Build cross-references: find which test files and docs mention each symbol
    let (test_refs, doc_refs) = build_cross_references(graph);

    // Extract usage examples from test files and documentation
    let usage_examples_map = extract_usage_examples(graph);

    // Build symbol table from graph symbols
    for symbol in &graph.symbols {
        let subsystem = symbol_to_subsystem
            .get(&symbol.name)
            .cloned()
            .unwrap_or_else(|| "unclustered".to_string());

        let related_symbols: Vec<String> = related_map
            .get(&symbol.name)
            .map(|set| set.iter().cloned().collect())
            .unwrap_or_default();

        let signature = build_signature(symbol);
        let chunk = build_chunk_reference(symbol);
        let summary = build_symbol_summary(symbol);

        // Get file from first chunk_id
        let file = symbol
            .chunk_ids
            .first()
            .and_then(|cid| chunk_to_file.get(cid))
            .cloned()
            .unwrap_or_default();

        // Get cross-references for this symbol
        let used_in_tests = test_refs.get(&symbol.name).cloned();
        let mentioned_in_docs = doc_refs.get(&symbol.name).cloned();
        let usage_examples = usage_examples_map.get(&symbol.name).cloned();

        symbols.insert(
            symbol.name.clone(),
            SymbolEntry {
                kind: symbol.kind.clone(),
                file,
                chunk,
                subsystem,
                signature,
                summary,
                related_symbols,
                embedding_index: None, // Could be populated if we track symbol->chunk->embedding mapping
                used_in_tests,
                mentioned_in_docs,
                usage_examples,
                llm_summary: None, // Will be populated by build_agent_index_with_llm if LLM engine is provided
            },
        );
    }

    symbols
}

/// Build a human-readable signature for a symbol
fn build_signature(symbol: &SymbolNode) -> Option<String> {
    match symbol.kind.as_str() {
        "function" | "method" => {
            let params = symbol
                .parameters
                .as_ref()
                .map(|p| {
                    p.iter()
                        .map(|(name, ty)| format!("{}: {}", name, ty))
                        .collect::<Vec<_>>()
                        .join(", ")
                })
                .unwrap_or_default();

            let ret = symbol
                .return_type
                .as_ref()
                .map(|r| format!(" -> {}", r))
                .unwrap_or_default();

            let generics = symbol
                .generics
                .as_ref()
                .map(|g| format!("<{}>", g))
                .unwrap_or_default();

            Some(format!("fn {}{}({}){}", symbol.name, generics, params, ret))
        }
        "struct" => {
            let generics = symbol
                .generics
                .as_ref()
                .map(|g| format!("<{}>", g))
                .unwrap_or_default();
            Some(format!("struct {}{}", symbol.name, generics))
        }
        "enum" => {
            let generics = symbol
                .generics
                .as_ref()
                .map(|g| format!("<{}>", g))
                .unwrap_or_default();
            Some(format!("enum {}{}", symbol.name, generics))
        }
        "trait" => {
            let generics = symbol
                .generics
                .as_ref()
                .map(|g| format!("<{}>", g))
                .unwrap_or_default();
            Some(format!("trait {}{}", symbol.name, generics))
        }
        "type_alias" => Some(format!("type {}", symbol.name)),
        _ => None,
    }
}

/// Build chunk reference string (byte range or line range)
fn build_chunk_reference(symbol: &SymbolNode) -> String {
    if let Some((start, end)) = symbol.byte_range {
        format!("{}-{}", start, end)
    } else if let Some((start_line, _)) = symbol.start_pos {
        if let Some((end_line, _)) = symbol.end_pos {
            format!("L{}-L{}", start_line, end_line)
        } else {
            format!("L{}", start_line)
        }
    } else {
        "unknown".to_string()
    }
}

/// Build a summary description for a symbol
fn build_symbol_summary(symbol: &SymbolNode) -> String {
    // Use docs if available
    if let Some(docs) = &symbol.docs {
        // Take first sentence or first 200 chars
        let first_line = docs.lines().next().unwrap_or(docs);
        if first_line.len() > 200 {
            format!("{}...", &first_line[..200])
        } else {
            first_line.to_string()
        }
    } else {
        // Generate generic summary based on kind
        match symbol.kind.as_str() {
            "function" | "method" => format!("Function {}", symbol.name),
            "struct" => format!("Struct {}", symbol.name),
            "enum" => format!("Enum {}", symbol.name),
            "trait" => format!("Trait {}", symbol.name),
            "impl" => format!("Implementation block for {}", symbol.name),
            "mod" => format!("Module {}", symbol.name),
            _ => format!("{} {}", symbol.kind, symbol.name),
        }
    }
}

/// Generate task-oriented views: pre-baked entry points for common agent tasks
fn generate_task_views(
    graph: &ProjectGraph,
    symbols: &HashMap<String, SymbolEntry>,
) -> HashMap<String, Vec<String>> {
    let mut tasks = HashMap::new();

    // Task: Add CLI command
    let cli_related: Vec<String> = symbols
        .iter()
        .filter(|(name, entry)| {
            (entry.kind == "enum" || entry.kind == "struct")
                && (name.to_lowercase().contains("command")
                    || name.to_lowercase().contains("cli")
                    || name.to_lowercase().contains("args"))
        })
        .map(|(_, entry)| format!("{}:{}", entry.file, entry.chunk))
        .collect();
    if !cli_related.is_empty() {
        tasks.insert("add_cli_command".to_string(), cli_related);
    }

    // Task: Modify RPC handler
    let rpc_handlers: Vec<String> = symbols
        .iter()
        .filter(|(name, entry)| {
            entry.kind == "function"
                && (name.contains("handle_")
                    || name.contains("process_")
                    || name.to_lowercase().contains("server"))
        })
        .map(|(_, entry)| format!("{}:{}", entry.file, entry.chunk))
        .collect();
    if !rpc_handlers.is_empty() {
        tasks.insert("modify_rpc_handler".to_string(), rpc_handlers);
    }

    // Task: Change docpack format
    let docpack_core: Vec<String> = symbols
        .iter()
        .filter(|(name, _)| {
            name.contains("Docpack") || name.contains("Manifest") || name.contains("Builder")
        })
        .map(|(_, entry)| format!("{}:{}", entry.file, entry.chunk))
        .collect();
    if !docpack_core.is_empty() {
        tasks.insert("change_docpack_format".to_string(), docpack_core);
    }

    // Task: Improve search
    let search_related: Vec<String> = symbols
        .iter()
        .filter(|(name, entry)| {
            entry.kind == "function"
                && (name.contains("search") || name.contains("query") || name.contains("find"))
        })
        .map(|(_, entry)| format!("{}:{}", entry.file, entry.chunk))
        .collect();
    if !search_related.is_empty() {
        tasks.insert("improve_search".to_string(), search_related);
    }

    // Task: Add subsystem (new module/component)
    let main_files: Vec<String> = graph
        .files
        .iter()
        .filter(|f| f.path.contains("main.rs") || f.path.contains("lib.rs"))
        .map(|f| f.path.clone())
        .collect();
    if !main_files.is_empty() {
        tasks.insert("add_subsystem".to_string(), main_files);
    }

    // Task: Modify embedding/ML inference
    let ml_related: Vec<String> = symbols
        .iter()
        .filter(|(name, entry)| {
            name.to_lowercase().contains("embed")
                || name.to_lowercase().contains("model")
                || name.to_lowercase().contains("infer")
                || entry.file.contains("embedding")
                || entry.file.contains("nlp")
        })
        .map(|(_, entry)| format!("{}:{}", entry.file, entry.chunk))
        .collect();
    if !ml_related.is_empty() {
        tasks.insert("modify_ml_inference".to_string(), ml_related);
    }

    tasks
}

/// Build impact graph: which symbols depend on which
fn build_impact_graph(graph: &ProjectGraph) -> HashMap<String, Vec<String>> {
    let mut impact = HashMap::new();

    for edge in &graph.edges {
        match edge.kind {
            GraphEdgeKind::SymbolToSymbol | GraphEdgeKind::SymbolSimilarity => {
                impact
                    .entry(edge.src.clone())
                    .or_insert_with(Vec::new)
                    .push(edge.dst.clone());
            }
            _ => {}
        }
    }

    // Sort and deduplicate
    for (_, targets) in impact.iter_mut() {
        targets.sort();
        targets.dedup();
    }

    impact
}

/// Generate quickstart navigation hints for agents
fn generate_quickstart(
    _graph: &ProjectGraph,
    subsystems: &[Subsystem],
    symbols: &HashMap<String, SymbolEntry>,
    impact_graph: &HashMap<String, Vec<String>>,
) -> AgentQuickstart {
    // Entry points: main functions, public API functions
    let entry_points: Vec<String> = symbols
        .iter()
        .filter(|(name, entry)| {
            name.as_str() == "main"
                || entry.kind == "function" && entry.file.contains("main.rs")
                || entry.kind == "function" && entry.file.contains("lib.rs")
        })
        .map(|(name, entry)| format!("{}:{}", entry.file, name))
        .take(5)
        .collect();

    // Core types: main structs/enums
    let core_types: Vec<String> = symbols
        .iter()
        .filter(|(_, entry)| {
            (entry.kind == "struct" || entry.kind == "enum")
                && !entry.file.contains("test")
                && !entry.file.contains("example")
        })
        .map(|(name, _)| name.clone())
        .take(10)
        .collect();

    // Most connected: symbols with most edges in impact graph
    let mut connected: Vec<(String, usize)> = impact_graph
        .iter()
        .map(|(symbol, targets)| (symbol.clone(), targets.len()))
        .collect();
    connected.sort_by(|a, b| b.1.cmp(&a.1));
    let most_connected: Vec<String> = connected
        .iter()
        .take(10)
        .map(|(symbol, _)| symbol.clone())
        .collect();

    // Subsystem map: task intent -> subsystem
    let mut subsystem_map = HashMap::new();
    for subsystem in subsystems {
        let role_lower = subsystem.role.to_lowercase();
        if role_lower.contains("cli") {
            subsystem_map.insert("if_adding_feature".to_string(), subsystem.name.clone());
        }
        if role_lower.contains("i/o")
            || role_lower.contains("archive")
            || role_lower.contains("docpack")
        {
            subsystem_map.insert("if_fixing_io".to_string(), subsystem.name.clone());
        }
        if role_lower.contains("server") || role_lower.contains("rpc") {
            subsystem_map.insert("if_changing_server".to_string(), subsystem.name.clone());
        }
        if role_lower.contains("parsing") || role_lower.contains("analysis") {
            subsystem_map.insert("if_modifying_parser".to_string(), subsystem.name.clone());
        }
        if role_lower.contains("embed") || role_lower.contains("ml") {
            subsystem_map.insert("if_changing_embeddings".to_string(), subsystem.name.clone());
        }
        if role_lower.contains("graph") {
            subsystem_map.insert("if_modifying_graph".to_string(), subsystem.name.clone());
        }
    }

    AgentQuickstart {
        entry_points,
        core_types,
        most_connected,
        subsystem_map,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_infer_subsystem_role() {
        let symbols = vec!["handle_request".to_string(), "ServerConfig".to_string()];
        let label = Some("server".to_string());
        let role = infer_subsystem_role(&symbols, &label);
        assert!(role.contains("server") || role.contains("RPC"));
    }

    #[test]
    fn test_build_signature() {
        let symbol = SymbolNode {
            name: "test_func".to_string(),
            kind: "function".to_string(),
            byte_range: Some((0, 100)),
            start_pos: Some((1, 0)),
            end_pos: Some((10, 0)),
            parent: None,
            docs: None,
            visibility: Some("pub".to_string()),
            parameters: Some(vec![("x".to_string(), "i32".to_string())]),
            return_type: Some("String".to_string()),
            generics: None,
            trait_impl: None,
            field_type: None,
            attributes: Vec::new(),
            is_mutable: false,
            modifiers: Vec::new(),
            chunk_ids: vec!["chunk_001".to_string()],
        };

        let sig = build_signature(&symbol);
        assert!(sig.is_some());
        assert!(sig.unwrap().contains("fn test_func"));
    }
}
