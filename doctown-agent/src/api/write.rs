// write.rs - Controlled write operations for code modification and generation
//
// These tools enable safe, structured modifications to code through MCP.
// All operations return proposed changes that can be reviewed before application.

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Request to apply a unified diff patch
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ApplyPatchRequest {
    pub file_path: String,
    pub patch: String,          // unified diff format
    pub validate: Option<bool>, // whether to validate syntax after applying
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ApplyPatchResponse {
    pub success: bool,
    pub file_path: String,
    pub changes: PatchChanges,
    pub validation: Option<ValidationResult>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PatchChanges {
    pub lines_added: usize,
    pub lines_removed: usize,
    pub hunks_applied: usize,
}

/// Request to propose a refactoring
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProposeRefactorRequest {
    pub symbol: String,
    pub refactor_type: RefactorType,
    pub options: HashMap<String, String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RefactorType {
    ExtractFunction,
    RenameSymbol,
    InlineFunction,
    ExtractVariable,
    MoveTrait,
    ChangeSignature,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProposeRefactorResponse {
    pub symbol: String,
    pub refactor_type: RefactorType,
    pub affected_files: Vec<String>,
    pub changes: Vec<FileChange>,
    pub impact_analysis: ImpactAnalysis,
    pub confidence: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FileChange {
    pub file_path: String,
    pub original: String,
    pub modified: String,
    pub change_type: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImpactAnalysis {
    pub symbols_affected: usize,
    pub files_affected: usize,
    pub dependencies_broken: Vec<String>,
    pub tests_to_update: Vec<String>,
}

/// Request to generate documentation for a symbol
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GenerateSymbolDocsRequest {
    pub symbol: String,
    pub style: Option<String>, // "rustdoc", "jsdoc", "pydoc", etc.
    pub include_examples: Option<bool>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GenerateSymbolDocsResponse {
    pub symbol: String,
    pub file_path: String,
    pub documentation: String,
    pub position: DocumentPosition,
    pub references_used: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DocumentPosition {
    pub line: usize,
    pub column: usize,
    pub insert_before: bool,
}

/// Request to rewrite a chunk with improved code
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RewriteChunkRequest {
    pub chunk_id: String,
    pub instructions: String,
    pub preserve_interface: Option<bool>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RewriteChunkResponse {
    pub chunk_id: String,
    pub file_path: String,
    pub original: String,
    pub rewritten: String,
    pub rationale: String,
    pub validation: ValidationResult,
}

/// Request to update a specific section of a file
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UpdateFileSectionRequest {
    pub file_path: String,
    pub start_line: usize,
    pub end_line: usize,
    pub new_content: String,
    pub preserve_formatting: Option<bool>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UpdateFileSectionResponse {
    pub file_path: String,
    pub lines_affected: usize,
    pub original_content: String,
    pub new_content: String,
    pub validation: ValidationResult,
}

/// Request to generate tests for a symbol
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CreateTestRequest {
    pub symbol: String,
    pub test_type: TestType,
    pub coverage_focus: Option<Vec<String>>, // specific scenarios to test
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TestType {
    Unit,
    Integration,
    PropertyBased,
    Benchmark,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CreateTestResponse {
    pub symbol: String,
    pub test_code: String,
    pub test_file_path: String,
    pub test_cases: Vec<TestCase>,
    pub setup_required: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TestCase {
    pub name: String,
    pub description: String,
    pub scenario: String,
}

/// Validation result for code changes
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationResult {
    pub valid: bool,
    pub errors: Vec<ValidationError>,
    pub warnings: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationError {
    pub line: usize,
    pub column: usize,
    pub message: String,
    pub severity: String,
}

// Implementation functions

/// Apply a unified diff patch to a file
pub fn apply_patch(
    request: &ApplyPatchRequest,
    symbols: &HashMap<String, doctown::docpack::SymbolEntry>,
    archive: &mut zip::ZipArchive<std::fs::File>,
) -> Result<ApplyPatchResponse> {
    // Read the original file
    let file_path = format!("source/{}", request.file_path.trim_start_matches('/'));
    let mut file = archive
        .by_name(&file_path)
        .context(format!("File not found: {}", request.file_path))?;

    let mut original_content = String::new();
    std::io::Read::read_to_string(&mut file, &mut original_content)?;

    // Parse and apply the patch
    let (modified_content, changes) = parse_and_apply_patch(&original_content, &request.patch)?;

    // Validate if requested
    let validation = if request.validate.unwrap_or(false) {
        Some(validate_content(&modified_content, &request.file_path)?)
    } else {
        None
    };

    Ok(ApplyPatchResponse {
        success: validation.as_ref().map(|v| v.valid).unwrap_or(true),
        file_path: request.file_path.clone(),
        changes,
        validation,
    })
}

/// Propose a refactoring with impact analysis
pub fn propose_refactor(
    request: &ProposeRefactorRequest,
    symbols: &HashMap<String, doctown::docpack::SymbolEntry>,
    graph: &doctown::docpack::GraphFile,
    _archive: &mut zip::ZipArchive<std::fs::File>,
) -> Result<ProposeRefactorResponse> {
    // Find the symbol
    let symbol_entry = symbols
        .get(&request.symbol)
        .ok_or_else(|| anyhow::anyhow!("Symbol not found: {}", request.symbol))?;

    // Analyze impact using the dependency graph
    let impact = analyze_refactor_impact(&request.symbol, graph, symbols)?;

    // Generate the proposed changes based on refactor type
    let changes = generate_refactor_changes(request, symbol_entry, symbols, _archive)?;

    // Calculate confidence based on complexity and impact
    let confidence = calculate_refactor_confidence(&changes, &impact);

    let affected_files: Vec<String> = changes.iter().map(|c| c.file_path.clone()).collect();

    Ok(ProposeRefactorResponse {
        symbol: request.symbol.clone(),
        refactor_type: request.refactor_type.clone(),
        affected_files,
        changes,
        impact_analysis: impact,
        confidence,
    })
}

/// Generate documentation for a symbol
pub fn generate_symbol_docs(
    request: &GenerateSymbolDocsRequest,
    symbols: &HashMap<String, doctown::docpack::SymbolEntry>,
    chunks: &[doctown::docpack::ChunkEntry],
    _archive: &mut zip::ZipArchive<std::fs::File>,
) -> Result<GenerateSymbolDocsResponse> {
    // Find the symbol
    let symbol_entry = symbols
        .get(&request.symbol)
        .ok_or_else(|| anyhow::anyhow!("Symbol not found: {}", request.symbol))?;

    // Get the symbol's context from chunks
    let context = gather_symbol_context(symbol_entry, chunks, symbols)?;

    // Generate documentation based on style
    let style = request.style.as_deref().unwrap_or("rustdoc");
    let documentation = generate_docs(&context, style, request.include_examples.unwrap_or(true))?;

    // Determine position for insertion
    let position = calculate_doc_position(symbol_entry)?;

    Ok(GenerateSymbolDocsResponse {
        symbol: request.symbol.clone(),
        file_path: symbol_entry.file.clone(),
        documentation,
        position,
        references_used: context.references,
    })
}

/// Rewrite a chunk with improvements
pub fn rewrite_chunk(
    request: &RewriteChunkRequest,
    chunks: &[doctown::docpack::ChunkEntry],
    symbols: &HashMap<String, doctown::docpack::SymbolEntry>,
) -> Result<RewriteChunkResponse> {
    // Find the chunk
    let chunk = chunks
        .iter()
        .find(|c| c.chunk_id == request.chunk_id)
        .ok_or_else(|| anyhow::anyhow!("Chunk not found: {}", request.chunk_id))?;

    // Analyze the chunk and generate improved version
    let (rewritten, rationale) = generate_improved_chunk(
        &chunk.text,
        &request.instructions,
        request.preserve_interface.unwrap_or(true),
        symbols,
    )?;

    // Validate the rewritten code
    let validation = validate_content(&rewritten, &chunk.file_path)?;

    Ok(RewriteChunkResponse {
        chunk_id: request.chunk_id.clone(),
        file_path: chunk.file_path.clone(),
        original: chunk.text.clone(),
        rewritten,
        rationale,
        validation,
    })
}

/// Update a specific section of a file
pub fn update_file_section(
    request: &UpdateFileSectionRequest,
    archive: &mut zip::ZipArchive<std::fs::File>,
) -> Result<UpdateFileSectionResponse> {
    // Read the original file
    let file_path = format!("source/{}", request.file_path.trim_start_matches('/'));
    let mut file = archive
        .by_name(&file_path)
        .context(format!("File not found: {}", request.file_path))?;

    let mut original_content = String::new();
    std::io::Read::read_to_string(&mut file, &mut original_content)?;

    // Extract the section being replaced
    let lines: Vec<&str> = original_content.lines().collect();
    let original_section = lines[(request.start_line - 1)..request.end_line].join("\n");

    // Apply formatting if requested
    let new_content = if request.preserve_formatting.unwrap_or(true) {
        preserve_indentation(&original_section, &request.new_content)
    } else {
        request.new_content.clone()
    };

    // Validate the new content
    let validation = validate_content(&new_content, &request.file_path)?;

    let lines_affected = request.end_line - request.start_line + 1;

    Ok(UpdateFileSectionResponse {
        file_path: request.file_path.clone(),
        lines_affected,
        original_content: original_section,
        new_content,
        validation,
    })
}

/// Generate tests for a symbol
pub fn create_test_for_symbol(
    request: &CreateTestRequest,
    symbols: &HashMap<String, doctown::docpack::SymbolEntry>,
    chunks: &[doctown::docpack::ChunkEntry],
    _archive: &mut zip::ZipArchive<std::fs::File>,
) -> Result<CreateTestResponse> {
    // Find the symbol
    let symbol_entry = symbols
        .get(&request.symbol)
        .ok_or_else(|| anyhow::anyhow!("Symbol not found: {}", request.symbol))?;

    // Gather context about the symbol
    let context = gather_symbol_context(symbol_entry, chunks, symbols)?;

    // Generate test cases based on symbol type and coverage focus
    let test_cases = generate_test_cases(
        &context,
        &request.test_type,
        request.coverage_focus.as_ref(),
    )?;

    // Generate test code
    let test_code = generate_test_code(&request.symbol, &test_cases, &request.test_type)?;

    // Determine test file path
    let test_file_path = derive_test_path(&symbol_entry.file, &request.test_type)?;

    // Identify setup requirements
    let setup_required = identify_test_setup(&context, &test_cases)?;

    Ok(CreateTestResponse {
        symbol: request.symbol.clone(),
        test_code,
        test_file_path,
        test_cases,
        setup_required,
    })
}

// Helper structures and functions

#[derive(Debug)]
struct SymbolContext {
    signature: String,
    body: String,
    dependencies: Vec<String>,
    references: Vec<String>,
    doc_comment: Option<String>,
}

fn parse_and_apply_patch(original: &str, patch: &str) -> Result<(String, PatchChanges)> {
    // Simple patch parser - in production, use a proper diff library
    // This is a stub implementation that just returns the original for now

    let hunks_applied = patch.lines().filter(|l| l.starts_with("@@")).count();
    let lines_added = patch
        .lines()
        .filter(|l| l.starts_with("+") && !l.starts_with("+++"))
        .count();
    let lines_removed = patch
        .lines()
        .filter(|l| l.starts_with("-") && !l.starts_with("---"))
        .count();

    // TODO: Actually apply the patch - for now just return original
    let modified = original.to_string();

    Ok((
        modified,
        PatchChanges {
            lines_added,
            lines_removed,
            hunks_applied,
        },
    ))
}

fn validate_content(content: &str, _file_path: &str) -> Result<ValidationResult> {
    // Basic syntax validation - in production, integrate with language-specific parsers
    let mut errors = Vec::new();
    let mut warnings = Vec::new();

    // Check for common syntax issues
    if content.is_empty() {
        warnings.push("File content is empty".to_string());
    }

    // Check for unbalanced braces (simple heuristic)
    let open_braces = content.chars().filter(|&c| c == '{').count();
    let close_braces = content.chars().filter(|&c| c == '}').count();

    if open_braces != close_braces {
        errors.push(ValidationError {
            line: 0,
            column: 0,
            message: format!(
                "Unbalanced braces: {} open, {} close",
                open_braces, close_braces
            ),
            severity: "error".to_string(),
        });
    }

    let valid = errors.is_empty();

    Ok(ValidationResult {
        valid,
        errors,
        warnings,
    })
}

fn analyze_refactor_impact(
    symbol: &str,
    graph: &doctown::docpack::GraphFile,
    symbols: &HashMap<String, doctown::docpack::SymbolEntry>,
) -> Result<ImpactAnalysis> {
    // Find all edges involving this symbol
    let mut symbols_affected = std::collections::HashSet::new();
    let mut files_affected = std::collections::HashSet::new();

    for edge in &graph.edges {
        if edge.from == symbol || edge.to == symbol {
            symbols_affected.insert(edge.from.clone());
            symbols_affected.insert(edge.to.clone());

            if let Some(entry) = symbols.get(&edge.from) {
                files_affected.insert(entry.file.clone());
            }
            if let Some(entry) = symbols.get(&edge.to) {
                files_affected.insert(entry.file.clone());
            }
        }
    }

    Ok(ImpactAnalysis {
        symbols_affected: symbols_affected.len(),
        files_affected: files_affected.len(),
        dependencies_broken: Vec::new(), // TODO: Implement dependency analysis
        tests_to_update: Vec::new(),     // TODO: Identify related tests
    })
}

fn generate_refactor_changes(
    request: &ProposeRefactorRequest,
    symbol_entry: &doctown::docpack::SymbolEntry,
    _symbols: &HashMap<String, doctown::docpack::SymbolEntry>,
    _archive: &mut zip::ZipArchive<std::fs::File>,
) -> Result<Vec<FileChange>> {
    // Generate changes based on refactor type
    let mut changes = Vec::new();

    match request.refactor_type {
        RefactorType::RenameSymbol => {
            if let Some(new_name) = request.options.get("new_name") {
                changes.push(FileChange {
                    file_path: symbol_entry.file.clone(),
                    original: format!("fn {}(", request.symbol),
                    modified: format!("fn {}(", new_name),
                    change_type: "rename".to_string(),
                });
            }
        }
        RefactorType::ExtractFunction => {
            // TODO: Implement function extraction logic
            changes.push(FileChange {
                file_path: symbol_entry.file.clone(),
                original: "// extracted code".to_string(),
                modified: "// new function call".to_string(),
                change_type: "extract".to_string(),
            });
        }
        _ => {
            // Stub for other refactor types
        }
    }

    Ok(changes)
}

fn calculate_refactor_confidence(changes: &[FileChange], impact: &ImpactAnalysis) -> f32 {
    // Simple heuristic: lower confidence for larger impact
    let base = 0.9;
    let file_penalty = 0.05 * (impact.files_affected as f32).min(10.0);
    let symbol_penalty = 0.02 * (impact.symbols_affected as f32).min(20.0);
    let change_penalty = 0.03 * (changes.len() as f32).min(10.0);

    (base - file_penalty - symbol_penalty - change_penalty).max(0.1)
}

fn gather_symbol_context(
    symbol_entry: &doctown::docpack::SymbolEntry,
    chunks: &[doctown::docpack::ChunkEntry],
    symbols: &HashMap<String, doctown::docpack::SymbolEntry>,
) -> Result<SymbolContext> {
    // Find chunks containing this symbol
    let related_chunks: Vec<_> = chunks
        .iter()
        .filter(|c| c.file_path == symbol_entry.file)
        .collect();

    // Extract signature and body
    let signature = symbol_entry
        .signature
        .clone()
        .unwrap_or_else(|| "unknown signature".to_string());
    let body = related_chunks
        .first()
        .map(|c| c.text.clone())
        .unwrap_or_default();

    // Find dependencies from symbols in the same file
    let dependencies: Vec<String> = symbols
        .iter()
        .filter(|(_, s)| s.file == symbol_entry.file)
        .map(|(name, _)| name.clone())
        .take(5)
        .collect();

    Ok(SymbolContext {
        signature,
        body,
        dependencies,
        references: Vec::new(),
        doc_comment: None,
    })
}

fn generate_docs(_context: &SymbolContext, style: &str, include_examples: bool) -> Result<String> {
    let mut doc = String::new();

    match style {
        "rustdoc" => {
            doc.push_str("/// ");
            doc.push_str("TODO: Add description\n");
            doc.push_str("///\n");

            if include_examples {
                doc.push_str("/// # Examples\n");
                doc.push_str("///\n");
                doc.push_str("/// ```\n");
                doc.push_str("/// // Example usage\n");
                doc.push_str("/// ```\n");
            }
        }
        _ => {
            doc.push_str("// TODO: Add documentation\n");
        }
    }

    Ok(doc)
}

fn calculate_doc_position(
    _symbol_entry: &doctown::docpack::SymbolEntry,
) -> Result<DocumentPosition> {
    // Simplified - should parse file to find exact position
    Ok(DocumentPosition {
        line: 1, // TODO: Parse to find actual line
        column: 0,
        insert_before: true,
    })
}

fn generate_improved_chunk(
    original: &str,
    instructions: &str,
    _preserve_interface: bool,
    _symbols: &HashMap<String, doctown::docpack::SymbolEntry>,
) -> Result<(String, String)> {
    // Stub: In production, this would use AI/rules to improve code
    let rewritten = format!("// Rewritten based on: {}\n{}", instructions, original);
    let rationale = format!("Applied instructions: {}", instructions);

    Ok((rewritten, rationale))
}

fn preserve_indentation(original: &str, new_content: &str) -> String {
    // Extract indentation from original
    let indent = original
        .chars()
        .take_while(|&c| c == ' ' || c == '\t')
        .collect::<String>();

    // Apply to new content
    new_content
        .lines()
        .map(|line| format!("{}{}", indent, line))
        .collect::<Vec<_>>()
        .join("\n")
}

fn generate_test_cases(
    _context: &SymbolContext,
    test_type: &TestType,
    coverage_focus: Option<&Vec<String>>,
) -> Result<Vec<TestCase>> {
    let mut cases = Vec::new();

    match test_type {
        TestType::Unit => {
            cases.push(TestCase {
                name: "test_basic_functionality".to_string(),
                description: "Tests basic functionality".to_string(),
                scenario: "Normal input case".to_string(),
            });
            cases.push(TestCase {
                name: "test_edge_cases".to_string(),
                description: "Tests edge cases".to_string(),
                scenario: "Boundary conditions".to_string(),
            });
        }
        TestType::Integration => {
            cases.push(TestCase {
                name: "test_integration".to_string(),
                description: "Tests integration with other components".to_string(),
                scenario: "End-to-end flow".to_string(),
            });
        }
        _ => {
            // Other test types
        }
    }

    // Filter by coverage focus if provided
    if let Some(focus) = coverage_focus {
        cases.retain(|case| focus.iter().any(|f| case.name.contains(f)));
    }

    Ok(cases)
}

fn generate_test_code(symbol: &str, cases: &[TestCase], _test_type: &TestType) -> Result<String> {
    let mut code = String::new();

    code.push_str("#[cfg(test)]\n");
    code.push_str("mod tests {\n");
    code.push_str("    use super::*;\n\n");

    for case in cases {
        code.push_str(&format!("    #[test]\n"));
        code.push_str(&format!("    fn {}() {{\n", case.name));
        code.push_str(&format!("        // {}\n", case.description));
        code.push_str(&format!("        // TODO: Implement test for {}\n", symbol));
        code.push_str("        assert!(true);\n");
        code.push_str("    }\n\n");
    }

    code.push_str("}\n");

    Ok(code)
}

fn derive_test_path(source_path: &str, test_type: &TestType) -> Result<String> {
    let path_parts: Vec<&str> = source_path.split('/').collect();
    let filename = path_parts.last().unwrap_or(&"unknown.rs");
    let name_without_ext = filename.trim_end_matches(".rs");

    match test_type {
        TestType::Unit => Ok(format!("{}_test.rs", name_without_ext)),
        TestType::Integration => Ok(format!("tests/{}_integration.rs", name_without_ext)),
        _ => Ok(format!("tests/{}_{:?}.rs", name_without_ext, test_type)),
    }
}

fn identify_test_setup(context: &SymbolContext, _cases: &[TestCase]) -> Result<Vec<String>> {
    let mut setup = Vec::new();

    // Check if mocking is needed
    if !context.dependencies.is_empty() {
        setup.push("Mock dependencies".to_string());
    }

    // Check if test data is needed
    if context.body.contains("File") || context.body.contains("read") {
        setup.push("Test fixtures/data files".to_string());
    }

    Ok(setup)
}
