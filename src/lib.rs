// lib.rs - Doctown library

pub mod agent;
pub mod community;
pub mod docgen;
pub mod docpack;
pub mod embedding;
pub mod ingest;
pub mod nlp;

// Re-export commonly used types
pub use docpack::{AgentIndex, Subsystem, SymbolEntry, AgentQuickstart};
pub use ingest::{Embedding, ProcessedFile, ProjectGraph, SymbolNode, GraphEdge, GraphEdgeKind};
