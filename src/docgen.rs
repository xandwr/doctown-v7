// docgen.rs - Comprehensive documentation generation from project graph

use crate::ingest::{GraphEdge, GraphEdgeKind, ProcessedFile, SymbolNode};
use std::collections::{HashMap, HashSet};

/// Complete documentation structure generated from the project graph
#[derive(Debug, Clone)]
pub struct GeneratedDocs {
    pub module_summaries: Vec<ModuleSummary>,
    pub file_summaries: Vec<FileSummary>,
    pub struct_docs: Vec<StructDoc>,
    pub function_docs: Vec<FunctionDoc>,
    pub architecture_overview: ArchitectureOverview,
    pub dependency_overview: DependencyOverview,
    pub cluster_summaries: Vec<ClusterSummary>,
}

/// Module-level summary with aggregated stats
#[derive(Debug, Clone)]
pub struct ModuleSummary {
    pub module_path: String,
    pub description: String,
    pub file_count: usize,
    pub symbol_count: usize,
    pub lines_of_code: usize,
    pub primary_purpose: String,
    pub key_exports: Vec<String>,
    pub submodules: Vec<String>,
}

/// Per-file summary with detailed breakdown
#[derive(Debug, Clone)]
pub struct FileSummary {
    pub file_path: String,
    pub description: String,
    pub language: String,
    pub lines_total: usize,
    pub lines_code: usize,
    pub lines_comment: usize,
    pub lines_blank: usize,
    pub symbols: Vec<String>,
    pub imports: Vec<String>,
    pub exports: Vec<String>,
    pub dependencies: Vec<String>,
    pub complexity_score: f32,
}

/// Struct documentation with fields and relationships
#[derive(Debug, Clone)]
pub struct StructDoc {
    pub name: String,
    pub file_path: String,
    pub description: String,
    pub visibility: String,
    pub fields: Vec<FieldInfo>,
    pub methods: Vec<String>,
    pub traits_implemented: Vec<String>,
    pub derives: Vec<String>,
    pub related_structs: Vec<String>,
}

#[derive(Debug, Clone)]
pub struct FieldInfo {
    pub name: String,
    pub field_type: String,
    pub visibility: String,
    pub description: String,
}

/// Function documentation with signature and usage
#[derive(Debug, Clone)]
pub struct FunctionDoc {
    pub name: String,
    pub file_path: String,
    pub description: String,
    pub signature: String,
    pub parameters: Vec<ParamInfo>,
    pub return_type: String,
    pub visibility: String,
    pub is_async: bool,
    pub is_unsafe: bool,
    pub calls: Vec<String>,
    pub called_by: Vec<String>,
    pub complexity_estimate: String,
}

#[derive(Debug, Clone)]
pub struct ParamInfo {
    pub name: String,
    pub param_type: String,
    pub description: String,
}

/// High-level architecture overview
#[derive(Debug, Clone)]
pub struct ArchitectureOverview {
    pub total_files: usize,
    pub total_symbols: usize,
    pub total_lines: usize,
    pub language_breakdown: HashMap<String, usize>,
    pub module_hierarchy: String,
    pub core_components: Vec<CoreComponent>,
    pub design_patterns: Vec<String>,
    pub architectural_style: String,
}

#[derive(Debug, Clone)]
pub struct CoreComponent {
    pub name: String,
    pub purpose: String,
    pub key_files: Vec<String>,
    pub interaction_count: usize,
}

/// Dependency relationships and analysis
#[derive(Debug, Clone)]
pub struct DependencyOverview {
    pub internal_dependencies: Vec<DependencyEdge>,
    pub external_dependencies: Vec<ExternalDependency>,
    pub dependency_graph_summary: String,
    pub circular_dependencies: Vec<CircularDep>,
    pub most_depended_on: Vec<(String, usize)>,
    pub least_coupled: Vec<String>,
}

#[derive(Debug, Clone)]
pub struct DependencyEdge {
    pub from: String,
    pub to: String,
    pub dependency_type: String,
    pub strength: f32,
}

#[derive(Debug, Clone)]
pub struct ExternalDependency {
    pub name: String,
    pub version: Option<String>,
    pub used_by: Vec<String>,
}

#[derive(Debug, Clone)]
pub struct CircularDep {
    pub cycle: Vec<String>,
    pub severity: String,
}

/// Semantic cluster/topic summary
#[derive(Debug, Clone)]
pub struct ClusterSummary {
    pub cluster_id: String,
    pub topic_label: String,
    pub description: String,
    pub chunk_count: usize,
    pub symbol_count: usize,
    pub representative_symbols: Vec<String>,
    pub representative_chunks: Vec<String>,
    pub coherence_score: f32,
}

/// Main documentation generator
pub struct DocGenerator<'a> {
    processed_files: &'a [ProcessedFile],
    all_edges: Vec<GraphEdge>,
    symbol_map: HashMap<String, (&'a SymbolNode, String)>, // symbol_name -> (symbol, file_path)
}

impl<'a> DocGenerator<'a> {
    pub fn new(processed_files: &'a [ProcessedFile]) -> Self {
        // Collect all edges from processed files
        let all_edges: Vec<GraphEdge> = processed_files
            .iter()
            .flat_map(|pf| pf.graph_edges.clone())
            .collect();

        // Build symbol map for quick lookups
        let mut symbol_map = HashMap::new();
        for pf in processed_files {
            for symbol in &pf.symbols {
                let key = format!("{}::{}", pf.file_node.path, symbol.name);
                symbol_map.insert(key, (symbol, pf.file_node.path.clone()));
            }
        }

        Self {
            processed_files,
            all_edges,
            symbol_map,
        }
    }

    /// Generate all documentation from the graph
    pub fn generate_all(&self) -> GeneratedDocs {
        GeneratedDocs {
            module_summaries: self.generate_module_summaries(),
            file_summaries: self.generate_file_summaries(),
            struct_docs: self.generate_struct_docs(),
            function_docs: self.generate_function_docs(),
            architecture_overview: self.generate_architecture_overview(),
            dependency_overview: self.generate_dependency_overview(),
            cluster_summaries: self.generate_cluster_summaries(),
        }
    }

    /// Generate module-level summaries by grouping files
    fn generate_module_summaries(&self) -> Vec<ModuleSummary> {
        let mut modules: HashMap<String, Vec<&ProcessedFile>> = HashMap::new();

        // Group files by module (directory)
        for pf in self.processed_files {
            let module_path = if let Some(parent) = std::path::Path::new(&pf.file_node.path)
                .parent()
                .and_then(|p| p.to_str())
            {
                if parent.is_empty() {
                    "<root>".to_string()
                } else {
                    parent.to_string()
                }
            } else {
                "<root>".to_string()
            };

            modules.entry(module_path).or_default().push(pf);
        }

        let mut summaries = Vec::new();

        for (module_path, files) in modules {
            let symbol_count: usize = files.iter().map(|f| f.symbols.len()).sum();
            let lines_of_code: usize = files
                .iter()
                .map(|f| {
                    let text = String::from_utf8_lossy(&f.original_bytes);
                    text.lines().count()
                })
                .sum();

            // Determine primary purpose based on file types and symbols
            let primary_purpose = self.infer_module_purpose(&files);

            // Extract key public exports
            let key_exports = self.extract_key_exports(&files);

            // Find submodules
            let submodules = self.find_submodules(&module_path, &files);

            let description =
                self.generate_module_description(&module_path, &files, &primary_purpose);

            summaries.push(ModuleSummary {
                module_path: module_path.clone(),
                description,
                file_count: files.len(),
                symbol_count,
                lines_of_code,
                primary_purpose,
                key_exports,
                submodules,
            });
        }

        summaries.sort_by(|a, b| a.module_path.cmp(&b.module_path));
        summaries
    }

    /// Generate per-file summaries
    fn generate_file_summaries(&self) -> Vec<FileSummary> {
        let mut summaries = Vec::new();

        for pf in self.processed_files {
            let text = String::from_utf8_lossy(&pf.original_bytes);
            let lines: Vec<&str> = text.lines().collect();
            let lines_total = lines.len();

            // Calculate line statistics
            let (lines_code, lines_comment, lines_blank) =
                self.calculate_line_stats(&text, &pf.file_node.path);

            let language = pf
                .metadata
                .get("filetype")
                .or_else(|| pf.metadata.get("language"))
                .cloned()
                .unwrap_or_else(|| "unknown".to_string());

            let symbols: Vec<String> = pf
                .symbols
                .iter()
                .map(|s| format!("{}::{}", s.kind, s.name))
                .collect();

            let imports = self.extract_imports(pf);
            let exports = self.extract_exports(pf);
            let dependencies = self.extract_file_dependencies(pf);

            // Complexity score based on symbol count, lines, and edges
            let complexity_score = self.calculate_file_complexity(pf);

            let description = self.generate_file_description(pf);

            summaries.push(FileSummary {
                file_path: pf.file_node.path.clone(),
                description,
                language,
                lines_total,
                lines_code,
                lines_comment,
                lines_blank,
                symbols,
                imports,
                exports,
                dependencies,
                complexity_score,
            });
        }

        summaries
    }

    /// Generate struct-specific documentation
    fn generate_struct_docs(&self) -> Vec<StructDoc> {
        let mut struct_docs = Vec::new();

        for pf in self.processed_files {
            for symbol in &pf.symbols {
                if symbol.kind == "struct" {
                    let fields = self.extract_struct_fields(pf, symbol);
                    let methods = self.find_struct_methods(pf, symbol);
                    let traits_impl = self.find_trait_implementations(pf, symbol);
                    let derives = self.extract_derives(symbol);
                    let related_structs = self.find_related_structs(symbol);

                    let description = symbol.docs.clone().unwrap_or_else(|| {
                        self.generate_struct_description(symbol, &fields, &methods)
                    });

                    struct_docs.push(StructDoc {
                        name: symbol.name.clone(),
                        file_path: pf.file_node.path.clone(),
                        description,
                        visibility: symbol
                            .visibility
                            .clone()
                            .unwrap_or_else(|| "private".to_string()),
                        fields,
                        methods,
                        traits_implemented: traits_impl,
                        derives,
                        related_structs,
                    });
                }
            }
        }

        struct_docs
    }

    /// Generate function-specific documentation
    fn generate_function_docs(&self) -> Vec<FunctionDoc> {
        let mut function_docs = Vec::new();

        for pf in self.processed_files {
            for symbol in &pf.symbols {
                if symbol.kind == "function" {
                    let signature = self.build_function_signature(symbol);

                    let parameters: Vec<ParamInfo> = symbol
                        .parameters
                        .as_ref()
                        .map(|params| {
                            params
                                .iter()
                                .map(|(name, ptype)| ParamInfo {
                                    name: name.clone(),
                                    param_type: ptype.clone(),
                                    description: String::new(), // Could be extracted from docs
                                })
                                .collect()
                        })
                        .unwrap_or_else(Vec::new);

                    let is_async = symbol.modifiers.contains(&"async".to_string());
                    let is_unsafe = symbol.modifiers.contains(&"unsafe".to_string());

                    let calls = self.find_function_calls(pf, symbol);
                    let called_by = self.find_callers(symbol);

                    let complexity = self.estimate_function_complexity(symbol, &calls);

                    let description = symbol.docs.clone().unwrap_or_else(|| {
                        self.generate_function_description(symbol, &parameters, &calls)
                    });

                    function_docs.push(FunctionDoc {
                        name: symbol.name.clone(),
                        file_path: pf.file_node.path.clone(),
                        description,
                        signature,
                        parameters,
                        return_type: symbol
                            .return_type
                            .clone()
                            .unwrap_or_else(|| "()".to_string()),
                        visibility: symbol
                            .visibility
                            .clone()
                            .unwrap_or_else(|| "private".to_string()),
                        is_async,
                        is_unsafe,
                        calls,
                        called_by,
                        complexity_estimate: complexity,
                    });
                }
            }
        }

        function_docs
    }

    /// Generate high-level architecture overview
    fn generate_architecture_overview(&self) -> ArchitectureOverview {
        let total_files = self.processed_files.len();
        let total_symbols: usize = self.processed_files.iter().map(|pf| pf.symbols.len()).sum();
        let total_lines: usize = self
            .processed_files
            .iter()
            .map(|pf| String::from_utf8_lossy(&pf.original_bytes).lines().count())
            .sum();

        // Language breakdown
        let mut language_breakdown: HashMap<String, usize> = HashMap::new();
        for pf in self.processed_files {
            let lang = pf
                .metadata
                .get("filetype")
                .or_else(|| pf.metadata.get("language"))
                .cloned()
                .unwrap_or_else(|| "unknown".to_string());
            *language_breakdown.entry(lang).or_default() += 1;
        }

        let module_hierarchy = self.build_module_hierarchy();
        let core_components = self.identify_core_components();
        let design_patterns = self.detect_design_patterns();
        let architectural_style = self.infer_architectural_style();

        ArchitectureOverview {
            total_files,
            total_symbols,
            total_lines,
            language_breakdown,
            module_hierarchy,
            core_components,
            design_patterns,
            architectural_style,
        }
    }

    /// Generate dependency overview with internal and external dependencies
    fn generate_dependency_overview(&self) -> DependencyOverview {
        let internal_dependencies = self.extract_internal_dependencies();
        let external_dependencies = self.extract_external_dependencies();
        let circular_dependencies = self.detect_circular_dependencies(&internal_dependencies);
        let most_depended_on = self.calculate_dependency_popularity(&internal_dependencies);
        let least_coupled = self.find_least_coupled_modules();

        let dependency_graph_summary = format!(
            "{} internal dependencies, {} external dependencies, {} circular dependencies detected",
            internal_dependencies.len(),
            external_dependencies.len(),
            circular_dependencies.len()
        );

        DependencyOverview {
            internal_dependencies,
            external_dependencies,
            dependency_graph_summary,
            circular_dependencies,
            most_depended_on,
            least_coupled,
        }
    }

    /// Generate cluster/topic summaries from semantic embeddings
    fn generate_cluster_summaries(&self) -> Vec<ClusterSummary> {
        // Extract cluster information from graph metadata
        let mut clusters: HashMap<String, Vec<String>> = HashMap::new();

        for pf in self.processed_files {
            for symbol in &pf.symbols {
                // Assuming cluster info would be in metadata
                // This is a placeholder - actual implementation would use embedding clusters
                let cluster_id = format!("cluster_{}", symbol.kind);
                clusters
                    .entry(cluster_id)
                    .or_default()
                    .push(symbol.name.clone());
            }
        }

        let mut summaries = Vec::new();

        for (cluster_id, symbols) in clusters {
            let topic_label = self.infer_topic_label(&symbols);
            let description = self.generate_cluster_description(&symbols);
            let representative_symbols = symbols.iter().take(5).cloned().collect();

            summaries.push(ClusterSummary {
                cluster_id: cluster_id.clone(),
                topic_label,
                description,
                chunk_count: 0, // Would be calculated from actual chunk clustering
                symbol_count: symbols.len(),
                representative_symbols,
                representative_chunks: Vec::new(),
                coherence_score: 0.85, // Placeholder
            });
        }

        summaries
    }

    // ===== Helper methods =====

    fn infer_module_purpose(&self, files: &[&ProcessedFile]) -> String {
        let symbol_kinds: Vec<String> = files
            .iter()
            .flat_map(|f| f.symbols.iter().map(|s| s.kind.clone()))
            .collect();

        if symbol_kinds.iter().any(|k| k == "trait") {
            "Interface definitions and abstractions".to_string()
        } else if symbol_kinds.iter().filter(|k| k == &"struct").count() > symbol_kinds.len() / 2 {
            "Data structures and types".to_string()
        } else if symbol_kinds.iter().filter(|k| k == &"function").count() > symbol_kinds.len() / 2
        {
            "Utility functions and operations".to_string()
        } else {
            "Mixed functionality module".to_string()
        }
    }

    fn extract_key_exports(&self, files: &[&ProcessedFile]) -> Vec<String> {
        let mut exports = Vec::new();
        for file in files {
            for symbol in &file.symbols {
                if let Some(vis) = &symbol.visibility {
                    if vis == "pub" {
                        exports.push(format!("{}::{}", symbol.kind, symbol.name));
                    }
                }
            }
        }
        exports.truncate(10); // Top 10 exports
        exports
    }

    fn find_submodules(&self, parent: &str, _files: &[&ProcessedFile]) -> Vec<String> {
        let mut submodules = HashSet::new();

        for pf in self.processed_files {
            if let Some(dir) = std::path::Path::new(&pf.file_node.path).parent() {
                let dir_str = dir.to_str().unwrap_or("");
                if dir_str.starts_with(parent) && dir_str != parent {
                    // Extract immediate subdirectory
                    let relative = dir_str.strip_prefix(parent).unwrap_or(dir_str);
                    if let Some(first_part) = relative.split('/').filter(|s| !s.is_empty()).next() {
                        submodules.insert(first_part.to_string());
                    }
                }
            }
        }

        submodules.into_iter().collect()
    }

    fn generate_module_description(
        &self,
        module_path: &str,
        files: &[&ProcessedFile],
        purpose: &str,
    ) -> String {
        format!(
            "Module '{}' contains {} file(s) implementing {}. Total symbols: {}",
            module_path,
            files.len(),
            purpose,
            files.iter().map(|f| f.symbols.len()).sum::<usize>()
        )
    }

    fn calculate_line_stats(&self, text: &str, file_path: &str) -> (usize, usize, usize) {
        let mut code = 0;
        let mut comment = 0;
        let mut blank = 0;

        let is_rust = file_path.ends_with(".rs");
        let mut in_multiline_comment = false;

        for line in text.lines() {
            let trimmed = line.trim();

            if trimmed.is_empty() {
                blank += 1;
                continue;
            }

            if is_rust {
                if trimmed.contains("/*") {
                    in_multiline_comment = true;
                }
                if in_multiline_comment {
                    comment += 1;
                    if trimmed.contains("*/") {
                        in_multiline_comment = false;
                    }
                    continue;
                }
                if trimmed.starts_with("//") {
                    comment += 1;
                    continue;
                }
            }

            code += 1;
        }

        (code, comment, blank)
    }

    fn extract_imports(&self, pf: &ProcessedFile) -> Vec<String> {
        let text = String::from_utf8_lossy(&pf.original_bytes);
        let mut imports = Vec::new();

        for line in text.lines() {
            let trimmed = line.trim();
            if trimmed.starts_with("use ") || trimmed.starts_with("import ") {
                imports.push(trimmed.to_string());
            }
        }

        imports.truncate(20); // Limit to top 20
        imports
    }

    fn extract_exports(&self, pf: &ProcessedFile) -> Vec<String> {
        pf.symbols
            .iter()
            .filter(|s| s.visibility.as_ref().map(|v| v == "pub").unwrap_or(false))
            .map(|s| s.name.clone())
            .collect()
    }

    fn extract_file_dependencies(&self, pf: &ProcessedFile) -> Vec<String> {
        self.all_edges
            .iter()
            .filter(|e| e.src == pf.file_node.path && matches!(e.kind, GraphEdgeKind::FileToFile))
            .map(|e| e.dst.clone())
            .collect()
    }

    fn calculate_file_complexity(&self, pf: &ProcessedFile) -> f32 {
        let symbol_count = pf.symbols.len() as f32;
        let chunk_count = pf.chunks.len() as f32;
        let edge_count = pf.graph_edges.len() as f32;

        (symbol_count * 0.4 + chunk_count * 0.3 + edge_count * 0.3) / 10.0
    }

    fn generate_file_description(&self, pf: &ProcessedFile) -> String {
        let symbol_breakdown = self.get_symbol_breakdown(&pf.symbols);
        format!(
            "File '{}' with {} symbols: {}",
            pf.file_node.path,
            pf.symbols.len(),
            symbol_breakdown
        )
    }

    fn get_symbol_breakdown(&self, symbols: &[SymbolNode]) -> String {
        let mut counts: HashMap<String, usize> = HashMap::new();
        for symbol in symbols {
            *counts.entry(symbol.kind.clone()).or_default() += 1;
        }

        counts
            .iter()
            .map(|(k, v)| format!("{} {}", v, k))
            .collect::<Vec<_>>()
            .join(", ")
    }

    fn extract_struct_fields(
        &self,
        pf: &ProcessedFile,
        struct_symbol: &SymbolNode,
    ) -> Vec<FieldInfo> {
        pf.symbols
            .iter()
            .filter(|s| s.kind == "field" && s.parent.as_ref() == Some(&struct_symbol.name))
            .map(|field| FieldInfo {
                name: field.name.clone(),
                field_type: field
                    .field_type
                    .clone()
                    .unwrap_or_else(|| "unknown".to_string()),
                visibility: field
                    .visibility
                    .clone()
                    .unwrap_or_else(|| "private".to_string()),
                description: field.docs.clone().unwrap_or_default(),
            })
            .collect()
    }

    fn find_struct_methods(&self, pf: &ProcessedFile, struct_symbol: &SymbolNode) -> Vec<String> {
        pf.symbols
            .iter()
            .filter(|s| s.kind == "function" && s.parent.as_ref() == Some(&struct_symbol.name))
            .map(|m| m.name.clone())
            .collect()
    }

    fn find_trait_implementations(
        &self,
        pf: &ProcessedFile,
        struct_symbol: &SymbolNode,
    ) -> Vec<String> {
        pf.symbols
            .iter()
            .filter(|s| {
                s.kind == "impl" && s.name.contains(&struct_symbol.name) && s.trait_impl.is_some()
            })
            .filter_map(|impl_block| impl_block.trait_impl.clone())
            .collect()
    }

    fn extract_derives(&self, symbol: &SymbolNode) -> Vec<String> {
        symbol
            .attributes
            .iter()
            .filter(|attr| attr.contains("derive"))
            .map(|attr| attr.clone())
            .collect()
    }

    fn find_related_structs(&self, _symbol: &SymbolNode) -> Vec<String> {
        // Placeholder - would analyze field types and method parameters
        Vec::new()
    }

    fn generate_struct_description(
        &self,
        symbol: &SymbolNode,
        fields: &[FieldInfo],
        methods: &[String],
    ) -> String {
        format!(
            "Struct '{}' with {} field(s) and {} method(s)",
            symbol.name,
            fields.len(),
            methods.len()
        )
    }

    fn build_function_signature(&self, symbol: &SymbolNode) -> String {
        let mut sig = String::new();

        if let Some(vis) = &symbol.visibility {
            sig.push_str(vis);
            sig.push(' ');
        }

        for modifier in &symbol.modifiers {
            sig.push_str(modifier);
            sig.push(' ');
        }

        sig.push_str("fn ");
        sig.push_str(&symbol.name);

        if let Some(generics) = &symbol.generics {
            sig.push_str(generics);
        }

        sig.push('(');
        if let Some(params) = &symbol.parameters {
            let param_strs: Vec<String> = params
                .iter()
                .map(|(name, ptype)| format!("{}: {}", name, ptype))
                .collect();
            sig.push_str(&param_strs.join(", "));
        }
        sig.push(')');

        if let Some(ret) = &symbol.return_type {
            sig.push_str(" -> ");
            sig.push_str(ret);
        }

        sig
    }

    fn find_function_calls(&self, _pf: &ProcessedFile, _symbol: &SymbolNode) -> Vec<String> {
        // Placeholder - would analyze symbol-to-symbol edges
        Vec::new()
    }

    fn find_callers(&self, _symbol: &SymbolNode) -> Vec<String> {
        // Placeholder - would analyze incoming symbol-to-symbol edges
        Vec::new()
    }

    fn estimate_function_complexity(&self, symbol: &SymbolNode, calls: &[String]) -> String {
        let param_count = symbol.parameters.as_ref().map(|p| p.len()).unwrap_or(0);
        let call_count = calls.len();

        let score = param_count + call_count * 2;

        if score < 5 {
            "Low".to_string()
        } else if score < 15 {
            "Medium".to_string()
        } else {
            "High".to_string()
        }
    }

    fn generate_function_description(
        &self,
        symbol: &SymbolNode,
        params: &[ParamInfo],
        _calls: &[String],
    ) -> String {
        format!(
            "Function '{}' with {} parameter(s)",
            symbol.name,
            params.len()
        )
    }

    fn build_module_hierarchy(&self) -> String {
        let mut modules = HashSet::new();

        for pf in self.processed_files {
            if let Some(parent) = std::path::Path::new(&pf.file_node.path).parent() {
                if let Some(parent_str) = parent.to_str() {
                    if !parent_str.is_empty() {
                        modules.insert(parent_str.to_string());
                    }
                }
            }
        }

        let mut sorted: Vec<_> = modules.into_iter().collect();
        sorted.sort();
        sorted.join(" → ")
    }

    fn identify_core_components(&self) -> Vec<CoreComponent> {
        // Identify files with high connectivity as core components
        let mut component_edges: HashMap<String, usize> = HashMap::new();

        for edge in &self.all_edges {
            *component_edges.entry(edge.src.clone()).or_default() += 1;
            *component_edges.entry(edge.dst.clone()).or_default() += 1;
        }

        let mut sorted: Vec<_> = component_edges.into_iter().collect();
        sorted.sort_by(|a, b| b.1.cmp(&a.1));

        sorted
            .iter()
            .take(5)
            .map(|(name, count)| CoreComponent {
                name: name.clone(),
                purpose: "Core component with high connectivity".to_string(),
                key_files: vec![name.clone()],
                interaction_count: *count,
            })
            .collect()
    }

    fn detect_design_patterns(&self) -> Vec<String> {
        let mut patterns = Vec::new();

        // Detect common patterns based on symbol analysis
        let has_traits = self
            .processed_files
            .iter()
            .any(|pf| pf.symbols.iter().any(|s| s.kind == "trait"));

        if has_traits {
            patterns.push("Trait-based abstraction".to_string());
        }

        let has_builders = self.processed_files.iter().any(|pf| {
            pf.symbols
                .iter()
                .any(|s| s.name.to_lowercase().contains("builder"))
        });

        if has_builders {
            patterns.push("Builder pattern".to_string());
        }

        patterns
    }

    fn infer_architectural_style(&self) -> String {
        let has_modules = self
            .processed_files
            .iter()
            .any(|pf| pf.symbols.iter().any(|s| s.kind == "mod"));

        if has_modules {
            "Modular architecture".to_string()
        } else {
            "Flat architecture".to_string()
        }
    }

    fn extract_internal_dependencies(&self) -> Vec<DependencyEdge> {
        self.all_edges
            .iter()
            .filter(|e| {
                matches!(
                    e.kind,
                    GraphEdgeKind::FileToFile | GraphEdgeKind::SymbolToSymbol
                )
            })
            .map(|e| DependencyEdge {
                from: e.src.clone(),
                to: e.dst.clone(),
                dependency_type: format!("{:?}", e.kind),
                strength: 1.0,
            })
            .collect()
    }

    fn extract_external_dependencies(&self) -> Vec<ExternalDependency> {
        // Placeholder - would parse Cargo.toml or similar
        Vec::new()
    }

    fn detect_circular_dependencies(&self, deps: &[DependencyEdge]) -> Vec<CircularDep> {
        // Simple cycle detection placeholder
        let mut cycles = Vec::new();

        for dep in deps {
            // Check if there's a reverse edge
            if deps.iter().any(|d| d.from == dep.to && d.to == dep.from) {
                cycles.push(CircularDep {
                    cycle: vec![dep.from.clone(), dep.to.clone(), dep.from.clone()],
                    severity: "Low".to_string(),
                });
            }
        }

        cycles.truncate(5); // Top 5 cycles
        cycles
    }

    fn calculate_dependency_popularity(&self, deps: &[DependencyEdge]) -> Vec<(String, usize)> {
        let mut popularity: HashMap<String, usize> = HashMap::new();

        for dep in deps {
            *popularity.entry(dep.to.clone()).or_default() += 1;
        }

        let mut sorted: Vec<_> = popularity.into_iter().collect();
        sorted.sort_by(|a, b| b.1.cmp(&a.1));
        sorted.truncate(10);
        sorted
    }

    fn find_least_coupled_modules(&self) -> Vec<String> {
        let mut edge_counts: HashMap<String, usize> = HashMap::new();

        for pf in self.processed_files {
            edge_counts.insert(pf.file_node.path.clone(), pf.graph_edges.len());
        }

        let mut sorted: Vec<_> = edge_counts.into_iter().collect();
        sorted.sort_by(|a, b| a.1.cmp(&b.1));

        sorted
            .iter()
            .take(5)
            .map(|(path, _)| path.clone())
            .collect()
    }

    fn infer_topic_label(&self, symbols: &[String]) -> String {
        // Simple topic inference based on common prefixes/terms
        if symbols.iter().any(|s| s.to_lowercase().contains("embed")) {
            "Embedding & Vector Processing".to_string()
        } else if symbols.iter().any(|s| s.to_lowercase().contains("parse")) {
            "Parsing & Analysis".to_string()
        } else if symbols.iter().any(|s| s.to_lowercase().contains("graph")) {
            "Graph & Relationships".to_string()
        } else {
            "General Functionality".to_string()
        }
    }

    fn generate_cluster_description(&self, symbols: &[String]) -> String {
        format!(
            "Cluster of {} related symbols with semantic similarity",
            symbols.len()
        )
    }
}
