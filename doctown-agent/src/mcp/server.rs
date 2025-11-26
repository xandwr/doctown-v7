// mcp/server.rs - MCP (Model Context Protocol) server with JSON-RPC 2.0 routing
//
// Exposes doctown agent API over stdio using MCP protocol.
// Clients can discover tools via list_tools and invoke them via call_tool.

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::io::{BufRead, BufReader, Write};
use std::path::PathBuf;
use zip::ZipArchive;

use crate::api;

/// JSON-RPC 2.0 request structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JsonRpcRequest {
    pub jsonrpc: String,
    pub id: Option<Value>,
    pub method: String,
    pub params: Option<Value>,
}

/// JSON-RPC 2.0 response structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JsonRpcResponse {
    pub jsonrpc: String,
    pub id: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub result: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<JsonRpcError>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JsonRpcError {
    pub code: i32,
    pub message: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub data: Option<Value>,
}

/// MCP Server state holding the loaded docpack
pub struct McpServer {
    docpack_path: PathBuf,
    agent_index: doctown::docpack::AgentIndex,
    chunks: Vec<doctown::docpack::ChunkEntry>,
    graph: doctown::docpack::GraphFile,
    embeddings: Vec<doctown::ingest::Embedding>,
    file_structure: Vec<doctown::docpack::FileStructureNode>,
}

impl McpServer {
    /// Create a new MCP server by loading a .docpack file
    pub fn new(docpack_path: PathBuf) -> Result<Self> {
        let file = std::fs::File::open(&docpack_path).context("Failed to open docpack file")?;

        let mut archive = ZipArchive::new(file).context("Failed to read docpack archive")?;

        // Load agent index
        let agent_index: doctown::docpack::AgentIndex = {
            let mut file = archive
                .by_name("agent_index.json")
                .context("agent_index.json not found in docpack")?;
            let mut content = String::new();
            std::io::Read::read_to_string(&mut file, &mut content)?;
            serde_json::from_str(&content).context("Failed to parse agent_index.json")?
        };

        // Load chunks
        let chunks: Vec<doctown::docpack::ChunkEntry> = {
            let mut file = archive
                .by_name("chunks.json")
                .context("chunks.json not found in docpack")?;
            let mut content = String::new();
            std::io::Read::read_to_string(&mut file, &mut content)?;
            serde_json::from_str(&content).context("Failed to parse chunks.json")?
        };

        // Load graph
        let graph: doctown::docpack::GraphFile = {
            let mut file = archive
                .by_name("graph.json")
                .context("graph.json not found in docpack")?;
            let mut content = String::new();
            std::io::Read::read_to_string(&mut file, &mut content)?;
            serde_json::from_str(&content).context("Failed to parse graph.json")?
        };

        // Load embeddings (stored as binary .bin file)
        let embeddings: Vec<doctown::ingest::Embedding> = {
            let mut file = archive
                .by_name("embeddings.bin")
                .context("embeddings.bin not found in docpack")?;
            let mut bytes = Vec::new();
            std::io::Read::read_to_end(&mut file, &mut bytes)?;

            // Parse binary embeddings (row-major f32 array)
            // We need to know the dimension - get it from manifest
            let manifest_file = std::fs::File::open(&docpack_path)?;
            let mut manifest_archive = ZipArchive::new(manifest_file)?;
            let manifest: doctown::docpack::Manifest = {
                let mut file = manifest_archive.by_name("manifest.json")?;
                let mut content = String::new();
                std::io::Read::read_to_string(&mut file, &mut content)?;
                serde_json::from_str(&content)?
            };

            let dim = manifest.embedding_dimensions;
            if dim == 0 {
                Vec::new()
            } else {
                let n_floats = bytes.len() / 4;
                let n_embeddings = n_floats / dim;
                let mut embeddings = Vec::with_capacity(n_embeddings);

                for i in 0..n_embeddings {
                    let start = i * dim * 4;
                    let end = start + dim * 4;
                    let chunk = &bytes[start..end];

                    let mut vec = Vec::with_capacity(dim);
                    for j in 0..dim {
                        let offset = j * 4;
                        let float_bytes = [
                            chunk[offset],
                            chunk[offset + 1],
                            chunk[offset + 2],
                            chunk[offset + 3],
                        ];
                        vec.push(f32::from_le_bytes(float_bytes));
                    }
                    embeddings.push(std::sync::Arc::from(vec.into_boxed_slice()));
                }
                embeddings
            }
        };

        // Load file structure
        let file_structure: Vec<doctown::docpack::FileStructureNode> = {
            let mut file = archive
                .by_name("filestructure.json")
                .context("filestructure.json not found in docpack")?;
            let mut content = String::new();
            std::io::Read::read_to_string(&mut file, &mut content)?;
            serde_json::from_str(&content).context("Failed to parse filestructure.json")?
        };

        Ok(Self {
            docpack_path,
            agent_index,
            chunks,
            graph,
            embeddings,
            file_structure,
        })
    }

    /// Start the MCP server, reading from stdin and writing to stdout
    pub fn run(&mut self) -> Result<()> {
        let stdin = std::io::stdin();
        let mut stdout = std::io::stdout();
        let reader = BufReader::new(stdin);

        for line in reader.lines() {
            let line = line?;
            if line.trim().is_empty() {
                continue;
            }

            let response = match serde_json::from_str::<JsonRpcRequest>(&line) {
                Ok(request) => self.handle_request(request),
                Err(e) => JsonRpcResponse {
                    jsonrpc: "2.0".to_string(),
                    id: None,
                    result: None,
                    error: Some(JsonRpcError {
                        code: -32700,
                        message: format!("Parse error: {}", e),
                        data: None,
                    }),
                },
            };

            let response_json = serde_json::to_string(&response)?;
            writeln!(stdout, "{}", response_json)?;
            stdout.flush()?;
        }

        Ok(())
    }

    /// Handle a JSON-RPC request (public for WebSocket wrapper)
    pub fn handle_request(&mut self, request: JsonRpcRequest) -> JsonRpcResponse {
        let result = match request.method.as_str() {
            "initialize" => self.handle_initialize(request.params),
            "tools/list" => self.handle_list_tools(),
            "tools/call" => self.handle_call_tool(request.params),
            _ => Err(anyhow::anyhow!("Method not found: {}", request.method)),
        };

        match result {
            Ok(value) => JsonRpcResponse {
                jsonrpc: "2.0".to_string(),
                id: request.id,
                result: Some(value),
                error: None,
            },
            Err(e) => JsonRpcResponse {
                jsonrpc: "2.0".to_string(),
                id: request.id,
                result: None,
                error: Some(JsonRpcError {
                    code: -32603,
                    message: e.to_string(),
                    data: None,
                }),
            },
        }
    }

    /// Handle initialize request
    fn handle_initialize(&self, _params: Option<Value>) -> Result<Value> {
        Ok(json!({
            "protocolVersion": "2024-11-05",
            "serverInfo": {
                "name": "doctown-agent",
                "version": "7.0.0"
            },
            "capabilities": {
                "tools": {}
            }
        }))
    }

    /// Handle tools/list request
    fn handle_list_tools(&self) -> Result<Value> {
        let tools = vec![
            json!({
                "name": "search_symbols",
                "description": "Search for symbols by name or content using keyword or semantic search",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "Search query"},
                        "limit": {"type": "number", "description": "Maximum results to return (default: 20)"},
                        "filter_subsystem": {"type": "string", "description": "Filter by subsystem name"},
                        "filter_kind": {"type": "string", "description": "Filter by symbol kind (function, struct, etc.)"}
                    },
                    "required": ["query"]
                }
            }),
            json!({
                "name": "get_symbol",
                "description": "Get detailed information about a specific symbol",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string", "description": "Symbol name"}
                    },
                    "required": ["name"]
                }
            }),
            json!({
                "name": "get_impact",
                "description": "Find all symbols that depend on the given symbol (forward impact analysis)",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "symbol": {"type": "string", "description": "Symbol name"},
                        "max_depth": {"type": "number", "description": "Maximum depth to traverse (default: 5)"}
                    },
                    "required": ["symbol"]
                }
            }),
            json!({
                "name": "get_dependencies",
                "description": "Find all symbols that the given symbol depends on (reverse impact analysis)",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "symbol": {"type": "string", "description": "Symbol name"},
                        "max_depth": {"type": "number", "description": "Maximum depth to traverse (default: 5)"}
                    },
                    "required": ["symbol"]
                }
            }),
            json!({
                "name": "find_path",
                "description": "Find the shortest path between two symbols in the dependency graph",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "from": {"type": "string", "description": "Source symbol"},
                        "to": {"type": "string", "description": "Target symbol"},
                        "max_depth": {"type": "number", "description": "Maximum path length (default: 10)"}
                    },
                    "required": ["from", "to"]
                }
            }),
            json!({
                "name": "list_subsystems",
                "description": "List all detected subsystems/modules with their confidence scores",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string", "description": "Filter by subsystem name"},
                        "min_confidence": {"type": "number", "description": "Minimum confidence score (0-1)"}
                    }
                }
            }),
            json!({
                "name": "get_subsystem",
                "description": "Get detailed information about a specific subsystem",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string", "description": "Subsystem name"}
                    },
                    "required": ["name"]
                }
            }),
            json!({
                "name": "get_quickstart",
                "description": "Get a quickstart guide with entry points, core types, and navigation hints",
                "inputSchema": {
                    "type": "object",
                    "properties": {}
                }
            }),
            json!({
                "name": "get_task_view",
                "description": "Get symbols relevant to a specific development task",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "task_type": {"type": "string", "description": "Task type (e.g., 'add_feature', 'fix_bug')"},
                        "context": {"type": "string", "description": "Additional context about the task"}
                    },
                    "required": ["task_type"]
                }
            }),
            json!({
                "name": "read_file",
                "description": "Read the complete content of a source file from the docpack",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string", "description": "File path"}
                    },
                    "required": ["path"]
                }
            }),
            json!({
                "name": "get_symbol_content",
                "description": "Get the source code content for a specific symbol",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "symbol": {"type": "string", "description": "Symbol name"}
                    },
                    "required": ["symbol"]
                }
            }),
            json!({
                "name": "list_files",
                "description": "List all files in the codebase",
                "inputSchema": {
                    "type": "object",
                    "properties": {}
                }
            }),
            // Write operations - controlled code modification
            json!({
                "name": "apply_patch",
                "description": "Apply a unified diff patch to a file with validation",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "file_path": {"type": "string", "description": "Path to the file to patch"},
                        "patch": {"type": "string", "description": "Unified diff format patch"},
                        "validate": {"type": "boolean", "description": "Whether to validate syntax after applying (default: false)"}
                    },
                    "required": ["file_path", "patch"]
                }
            }),
            json!({
                "name": "propose_refactor",
                "description": "Propose a refactoring with impact analysis and confidence score",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "symbol": {"type": "string", "description": "Symbol to refactor"},
                        "refactor_type": {
                            "type": "string",
                            "description": "Type of refactoring",
                            "enum": ["extract_function", "rename_symbol", "inline_function", "extract_variable", "move_trait", "change_signature"]
                        },
                        "options": {
                            "type": "object",
                            "description": "Refactoring options (e.g., new_name for rename)"
                        }
                    },
                    "required": ["symbol", "refactor_type"]
                }
            }),
            json!({
                "name": "generate_symbol_docs",
                "description": "Generate documentation for a symbol with examples",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "symbol": {"type": "string", "description": "Symbol to document"},
                        "style": {"type": "string", "description": "Documentation style (rustdoc, jsdoc, pydoc, etc.)"},
                        "include_examples": {"type": "boolean", "description": "Include usage examples (default: true)"}
                    },
                    "required": ["symbol"]
                }
            }),
            json!({
                "name": "rewrite_chunk",
                "description": "Rewrite a code chunk with improvements based on instructions",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "chunk_id": {"type": "string", "description": "ID of the chunk to rewrite"},
                        "instructions": {"type": "string", "description": "Instructions for rewriting"},
                        "preserve_interface": {"type": "boolean", "description": "Preserve public interface (default: true)"}
                    },
                    "required": ["chunk_id", "instructions"]
                }
            }),
            json!({
                "name": "update_file_section",
                "description": "Update a specific section of a file by line range",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "file_path": {"type": "string", "description": "Path to the file"},
                        "start_line": {"type": "number", "description": "Start line (1-indexed)"},
                        "end_line": {"type": "number", "description": "End line (inclusive)"},
                        "new_content": {"type": "string", "description": "New content for the section"},
                        "preserve_formatting": {"type": "boolean", "description": "Preserve indentation (default: true)"}
                    },
                    "required": ["file_path", "start_line", "end_line", "new_content"]
                }
            }),
            json!({
                "name": "create_test_for_symbol",
                "description": "Generate test cases and test code for a symbol",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "symbol": {"type": "string", "description": "Symbol to test"},
                        "test_type": {
                            "type": "string",
                            "description": "Type of test to generate",
                            "enum": ["unit", "integration", "property_based", "benchmark"]
                        },
                        "coverage_focus": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Specific scenarios to test"
                        }
                    },
                    "required": ["symbol", "test_type"]
                }
            }),
        ];

        Ok(json!({"tools": tools}))
    }

    /// Handle tools/call request
    fn handle_call_tool(&mut self, params: Option<Value>) -> Result<Value> {
        let params = params.ok_or_else(|| anyhow::anyhow!("Missing params"))?;
        let tool_name = params
            .get("name")
            .and_then(|v| v.as_str())
            .ok_or_else(|| anyhow::anyhow!("Missing tool name"))?;
        let arguments = params.get("arguments").cloned();

        let result = match tool_name {
            "search_symbols" => self.tool_search_symbols(arguments)?,
            "get_symbol" => self.tool_get_symbol(arguments)?,
            "get_impact" => self.tool_get_impact(arguments)?,
            "get_dependencies" => self.tool_get_dependencies(arguments)?,
            "find_path" => self.tool_find_path(arguments)?,
            "list_subsystems" => self.tool_list_subsystems(arguments)?,
            "get_subsystem" => self.tool_get_subsystem(arguments)?,
            "get_quickstart" => self.tool_get_quickstart()?,
            "get_task_view" => self.tool_get_task_view(arguments)?,
            "read_file" => self.tool_read_file(arguments)?,
            "get_symbol_content" => self.tool_get_symbol_content(arguments)?,
            "list_files" => self.tool_list_files()?,
            // Write operations
            "apply_patch" => self.tool_apply_patch(arguments)?,
            "propose_refactor" => self.tool_propose_refactor(arguments)?,
            "generate_symbol_docs" => self.tool_generate_symbol_docs(arguments)?,
            "rewrite_chunk" => self.tool_rewrite_chunk(arguments)?,
            "update_file_section" => self.tool_update_file_section(arguments)?,
            "create_test_for_symbol" => self.tool_create_test_for_symbol(arguments)?,
            _ => return Err(anyhow::anyhow!("Unknown tool: {}", tool_name)),
        };

        Ok(json!({
            "content": [{
                "type": "text",
                "text": serde_json::to_string_pretty(&result)?
            }]
        }))
    }

    // Tool implementations

    fn tool_search_symbols(&self, args: Option<Value>) -> Result<Value> {
        let query: api::search::SearchQuery =
            serde_json::from_value(args.ok_or_else(|| anyhow::anyhow!("Missing arguments"))?)?;
        let response = api::search::keyword_search(&query, &self.agent_index.symbols)?;
        Ok(serde_json::to_value(response)?)
    }

    fn tool_get_symbol(&self, args: Option<Value>) -> Result<Value> {
        let args = args.ok_or_else(|| anyhow::anyhow!("Missing arguments"))?;
        let name = args
            .get("name")
            .and_then(|v| v.as_str())
            .ok_or_else(|| anyhow::anyhow!("Missing symbol name"))?;
        let result = api::search::get_symbol(name, &self.agent_index.symbols)?;
        Ok(serde_json::to_value(result)?)
    }

    fn tool_get_impact(&self, args: Option<Value>) -> Result<Value> {
        let query: api::graph::ImpactQuery =
            serde_json::from_value(args.ok_or_else(|| anyhow::anyhow!("Missing arguments"))?)?;
        let response = api::graph::get_impact(&query, &self.agent_index.impact_graph)?;
        Ok(serde_json::to_value(response)?)
    }

    fn tool_get_dependencies(&self, args: Option<Value>) -> Result<Value> {
        let query: api::graph::DependencyQuery =
            serde_json::from_value(args.ok_or_else(|| anyhow::anyhow!("Missing arguments"))?)?;
        let response = api::graph::get_dependencies(&query, &self.graph)?;
        Ok(serde_json::to_value(response)?)
    }

    fn tool_find_path(&self, args: Option<Value>) -> Result<Value> {
        let query: api::graph::PathQuery =
            serde_json::from_value(args.ok_or_else(|| anyhow::anyhow!("Missing arguments"))?)?;
        let response = api::graph::find_path(&query, &self.graph)?;
        Ok(serde_json::to_value(response)?)
    }

    fn tool_list_subsystems(&self, args: Option<Value>) -> Result<Value> {
        let query: api::subsystems::SubsystemQuery = if let Some(a) = args {
            serde_json::from_value(a)?
        } else {
            api::subsystems::SubsystemQuery {
                name: None,
                min_confidence: None,
            }
        };
        let response = api::subsystems::list_subsystems(&query, &self.agent_index)?;
        Ok(serde_json::to_value(response)?)
    }

    fn tool_get_subsystem(&self, args: Option<Value>) -> Result<Value> {
        let args = args.ok_or_else(|| anyhow::anyhow!("Missing arguments"))?;
        let name = args
            .get("name")
            .and_then(|v| v.as_str())
            .ok_or_else(|| anyhow::anyhow!("Missing subsystem name"))?;
        let result = api::subsystems::get_subsystem_detail(name, &self.agent_index)?;
        Ok(serde_json::to_value(result)?)
    }

    fn tool_get_quickstart(&self) -> Result<Value> {
        let response = api::tasks::get_quickstart(&self.agent_index)?;
        Ok(serde_json::to_value(response)?)
    }

    fn tool_get_task_view(&self, args: Option<Value>) -> Result<Value> {
        let query: api::tasks::TaskQuery =
            serde_json::from_value(args.ok_or_else(|| anyhow::anyhow!("Missing arguments"))?)?;
        let response = api::tasks::get_task_view(&query, &self.agent_index)?;
        Ok(serde_json::to_value(response)?)
    }

    fn tool_read_file(&mut self, args: Option<Value>) -> Result<Value> {
        let request: api::editor::FileRequest =
            serde_json::from_value(args.ok_or_else(|| anyhow::anyhow!("Missing arguments"))?)?;

        let file = std::fs::File::open(&self.docpack_path)?;
        let mut archive = ZipArchive::new(file)?;

        let response = api::editor::read_file(&request, &mut archive)?;
        Ok(serde_json::to_value(response)?)
    }

    fn tool_get_symbol_content(&mut self, args: Option<Value>) -> Result<Value> {
        let request: api::editor::SymbolContentRequest =
            serde_json::from_value(args.ok_or_else(|| anyhow::anyhow!("Missing arguments"))?)?;

        let file = std::fs::File::open(&self.docpack_path)?;
        let mut archive = ZipArchive::new(file)?;

        let response =
            api::editor::get_symbol_content(&request, &self.agent_index.symbols, &mut archive)?;
        Ok(serde_json::to_value(response)?)
    }

    fn tool_list_files(&self) -> Result<Value> {
        let files = api::editor::list_files(&self.file_structure)?;
        Ok(serde_json::to_value(files)?)
    }

    // Write operation tools

    fn tool_apply_patch(&mut self, args: Option<Value>) -> Result<Value> {
        let request: api::write::ApplyPatchRequest =
            serde_json::from_value(args.ok_or_else(|| anyhow::anyhow!("Missing arguments"))?)?;

        let file = std::fs::File::open(&self.docpack_path)?;
        let mut archive = ZipArchive::new(file)?;

        let response = api::write::apply_patch(&request, &self.agent_index.symbols, &mut archive)?;
        Ok(serde_json::to_value(response)?)
    }

    fn tool_propose_refactor(&mut self, args: Option<Value>) -> Result<Value> {
        let request: api::write::ProposeRefactorRequest =
            serde_json::from_value(args.ok_or_else(|| anyhow::anyhow!("Missing arguments"))?)?;

        let file = std::fs::File::open(&self.docpack_path)?;
        let mut archive = ZipArchive::new(file)?;

        let response = api::write::propose_refactor(
            &request,
            &self.agent_index.symbols,
            &self.graph,
            &mut archive,
        )?;
        Ok(serde_json::to_value(response)?)
    }

    fn tool_generate_symbol_docs(&mut self, args: Option<Value>) -> Result<Value> {
        let request: api::write::GenerateSymbolDocsRequest =
            serde_json::from_value(args.ok_or_else(|| anyhow::anyhow!("Missing arguments"))?)?;

        let file = std::fs::File::open(&self.docpack_path)?;
        let mut archive = ZipArchive::new(file)?;

        let response = api::write::generate_symbol_docs(
            &request,
            &self.agent_index.symbols,
            &self.chunks,
            &mut archive,
        )?;
        Ok(serde_json::to_value(response)?)
    }

    fn tool_rewrite_chunk(&self, args: Option<Value>) -> Result<Value> {
        let request: api::write::RewriteChunkRequest =
            serde_json::from_value(args.ok_or_else(|| anyhow::anyhow!("Missing arguments"))?)?;

        let response =
            api::write::rewrite_chunk(&request, &self.chunks, &self.agent_index.symbols)?;
        Ok(serde_json::to_value(response)?)
    }

    fn tool_update_file_section(&mut self, args: Option<Value>) -> Result<Value> {
        let request: api::write::UpdateFileSectionRequest =
            serde_json::from_value(args.ok_or_else(|| anyhow::anyhow!("Missing arguments"))?)?;

        let file = std::fs::File::open(&self.docpack_path)?;
        let mut archive = ZipArchive::new(file)?;

        let response = api::write::update_file_section(&request, &mut archive)?;
        Ok(serde_json::to_value(response)?)
    }

    fn tool_create_test_for_symbol(&mut self, args: Option<Value>) -> Result<Value> {
        let request: api::write::CreateTestRequest =
            serde_json::from_value(args.ok_or_else(|| anyhow::anyhow!("Missing arguments"))?)?;

        let file = std::fs::File::open(&self.docpack_path)?;
        let mut archive = ZipArchive::new(file)?;

        let response = api::write::create_test_for_symbol(
            &request,
            &self.agent_index.symbols,
            &self.chunks,
            &mut archive,
        )?;
        Ok(serde_json::to_value(response)?)
    }
}
