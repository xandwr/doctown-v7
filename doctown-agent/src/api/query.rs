// query.rs - Conversational query interface for natural language questions about the codebase

use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConversationalQuery {
    pub question: String,
    pub limit: Option<usize>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConversationalAnswer {
    pub question: String,
    pub answer: String,
    pub relevant_symbols: Vec<RelevantSymbol>,
    pub relevant_files: Vec<String>,
    pub code_examples: Vec<CodeExample>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RelevantSymbol {
    pub name: String,
    pub kind: String,
    pub file: String,
    pub summary: String,
    pub relevance_score: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CodeExample {
    pub symbol: String,
    pub code: String,
    pub explanation: String,
}

/// Answer a natural language question about the codebase using embeddings + LLM
pub fn answer_question(
    query: &ConversationalQuery,
    symbols: &HashMap<String, doctown::docpack::SymbolEntry>,
    chunks: &[doctown::docpack::ChunkEntry],
    embeddings: &[Arc<[f32]>],
    embedding_engine: Option<&mut doctown::embedding::EmbeddingEngine>,
    llm_engine: Option<&doctown::llm::LlmEngine>,
) -> Result<ConversationalAnswer> {
    let limit = query.limit.unwrap_or(10);

    // Step 1: Embed the question
    let question_embedding = if let Some(engine) = embedding_engine {
        engine.embed(&query.question)?
    } else {
        // Fallback to keyword search if no embedding engine
        return answer_with_keyword_search(query, symbols, chunks, llm_engine);
    };

    // Step 2: Find top-K most relevant chunks via cosine similarity
    let mut scored_chunks: Vec<(usize, f32)> = embeddings
        .iter()
        .enumerate()
        .map(|(idx, emb)| {
            let score = cosine_similarity(&question_embedding, emb);
            (idx, score)
        })
        .collect();

    scored_chunks.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    scored_chunks.truncate(limit * 2); // Get more chunks than needed for filtering

    // Step 3: Map chunks back to symbols and aggregate
    let mut symbol_scores: HashMap<String, f32> = HashMap::new();
    let mut symbol_chunks: HashMap<String, Vec<String>> = HashMap::new();

    for (chunk_idx, score) in &scored_chunks {
        if *chunk_idx >= chunks.len() {
            continue;
        }
        let chunk = &chunks[*chunk_idx];

        // Find symbols that reference this chunk
        for (symbol_name, symbol_entry) in symbols.iter() {
            if symbol_entry.file == chunk.file_path {
                // Simple heuristic: if the chunk overlaps with symbol location
                *symbol_scores.entry(symbol_name.clone()).or_insert(0.0) += score;
                symbol_chunks
                    .entry(symbol_name.clone())
                    .or_insert_with(Vec::new)
                    .push(chunk.text.clone());
            }
        }
    }

    // Step 4: Sort symbols by relevance
    let mut relevant_symbols: Vec<RelevantSymbol> = symbol_scores
        .iter()
        .filter_map(|(name, score)| {
            symbols.get(name).map(|entry| RelevantSymbol {
                name: name.clone(),
                kind: entry.kind.clone(),
                file: entry.file.clone(),
                summary: entry.summary.clone(),
                relevance_score: *score,
            })
        })
        .collect();

    relevant_symbols.sort_by(|a, b| {
        b.relevance_score
            .partial_cmp(&a.relevance_score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    relevant_symbols.truncate(limit);

    // Step 5: Collect relevant files
    let mut relevant_files: Vec<String> = relevant_symbols
        .iter()
        .map(|s| s.file.clone())
        .collect::<std::collections::HashSet<_>>()
        .into_iter()
        .collect();
    relevant_files.sort();

    // Step 6: Collect code examples from top symbols
    let code_examples: Vec<CodeExample> = relevant_symbols
        .iter()
        .take(3) // Top 3 symbols
        .filter_map(|sym| {
            symbol_chunks.get(&sym.name).and_then(|chunks| {
                chunks.first().map(|code| CodeExample {
                    symbol: sym.name.clone(),
                    code: code.clone(),
                    explanation: sym.summary.clone(),
                })
            })
        })
        .collect();

    // Step 7: Generate natural language answer using LLM
    let answer = if let Some(llm) = llm_engine {
        generate_answer_with_llm(llm, query, &relevant_symbols, &code_examples)?
    } else {
        generate_answer_without_llm(query, &relevant_symbols, &code_examples)
    };

    Ok(ConversationalAnswer {
        question: query.question.clone(),
        answer,
        relevant_symbols,
        relevant_files,
        code_examples,
    })
}

/// Fallback to keyword-based search when embeddings are not available
fn answer_with_keyword_search(
    query: &ConversationalQuery,
    symbols: &HashMap<String, doctown::docpack::SymbolEntry>,
    chunks: &[doctown::docpack::ChunkEntry],
    llm_engine: Option<&doctown::llm::LlmEngine>,
) -> Result<ConversationalAnswer> {
    let query_lower = query.question.to_lowercase();
    let limit = query.limit.unwrap_or(10);

    // Simple keyword matching
    let mut relevant_symbols: Vec<RelevantSymbol> = symbols
        .iter()
        .filter_map(|(name, entry)| {
            let name_lower = name.to_lowercase();
            let summary_lower = entry.summary.to_lowercase();

            let mut score = 0.0;
            if name_lower.contains(&query_lower) {
                score += 10.0;
            }
            if summary_lower.contains(&query_lower) {
                score += 5.0;
            }

            if score > 0.0 {
                Some(RelevantSymbol {
                    name: name.clone(),
                    kind: entry.kind.clone(),
                    file: entry.file.clone(),
                    summary: entry.summary.clone(),
                    relevance_score: score,
                })
            } else {
                None
            }
        })
        .collect();

    relevant_symbols.sort_by(|a, b| {
        b.relevance_score
            .partial_cmp(&a.relevance_score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    relevant_symbols.truncate(limit);

    let relevant_files: Vec<String> = relevant_symbols
        .iter()
        .map(|s| s.file.clone())
        .collect::<std::collections::HashSet<_>>()
        .into_iter()
        .collect();

    // Find code examples from chunks
    let code_examples: Vec<CodeExample> = relevant_symbols
        .iter()
        .take(3)
        .filter_map(|sym| {
            chunks
                .iter()
                .find(|c| c.file_path == sym.file)
                .map(|chunk| CodeExample {
                    symbol: sym.name.clone(),
                    code: chunk.text.clone(),
                    explanation: sym.summary.clone(),
                })
        })
        .collect();

    let answer = if let Some(llm) = llm_engine {
        generate_answer_with_llm(llm, query, &relevant_symbols, &code_examples)?
    } else {
        generate_answer_without_llm(query, &relevant_symbols, &code_examples)
    };

    Ok(ConversationalAnswer {
        question: query.question.clone(),
        answer,
        relevant_symbols,
        relevant_files,
        code_examples,
    })
}

/// Generate a conversational answer using the LLM
fn generate_answer_with_llm(
    llm: &doctown::llm::LlmEngine,
    query: &ConversationalQuery,
    relevant_symbols: &[RelevantSymbol],
    code_examples: &[CodeExample],
) -> Result<String> {
    // Build context from relevant symbols
    let mut context = String::new();
    context.push_str("Relevant symbols found in the codebase:\n\n");

    for (i, sym) in relevant_symbols.iter().take(5).enumerate() {
        context.push_str(&format!(
            "{}. {} ({})\n   File: {}\n   Summary: {}\n\n",
            i + 1,
            sym.name,
            sym.kind,
            sym.file,
            sym.summary
        ));
    }

    if !code_examples.is_empty() {
        context.push_str("\nCode examples:\n\n");
        for example in code_examples.iter().take(2) {
            context.push_str(&format!(
                "From {}:\n```\n{}\n```\n\n",
                example.symbol,
                if example.code.len() > 500 {
                    &example.code[..500]
                } else {
                    &example.code
                }
            ));
        }
    }

    // Build prompt for LLM
    let prompt = format!(
        "You are a helpful coding assistant explaining a codebase to a developer.\n\n\
        Question: {}\n\n\
        {}\
        Based on the above information, provide a clear, concise answer to the question. \
        Include specific symbol names, file paths, and practical guidance. \
        Keep your answer to 2-3 paragraphs.",
        query.question, context
    );

    // Generate answer using LLM
    let answer = llm.explain_symbol(
        "codebase_query",
        "answer",
        None,
        Some(&prompt),
        "Answering question about codebase",
    )?;

    Ok(answer)
}

/// Generate a simple answer without LLM (fallback)
fn generate_answer_without_llm(
    query: &ConversationalQuery,
    relevant_symbols: &[RelevantSymbol],
    code_examples: &[CodeExample],
) -> String {
    let mut answer = format!("Based on your question: \"{}\"\n\n", query.question);

    if relevant_symbols.is_empty() {
        answer.push_str("I couldn't find any directly relevant symbols in the codebase. Try rephrasing your question or searching for specific terms.");
        return answer;
    }

    answer.push_str("Here's what I found:\n\n");

    for (i, sym) in relevant_symbols.iter().take(5).enumerate() {
        answer.push_str(&format!(
            "{}. **{}** ({})\n   - File: `{}`\n   - {}\n\n",
            i + 1,
            sym.name,
            sym.kind,
            sym.file,
            sym.summary
        ));
    }

    if !code_examples.is_empty() {
        answer.push_str("\nRelevant code examples:\n\n");
        for example in code_examples.iter().take(2) {
            answer.push_str(&format!(
                "From `{}`:\n```\n{}\n```\n\n",
                example.symbol,
                if example.code.len() > 300 {
                    format!("{}...", &example.code[..300])
                } else {
                    example.code.clone()
                }
            ));
        }
    }

    answer.push_str("\nFor more details, examine the specific symbols and files listed above.");
    answer
}

/// Calculate cosine similarity between two embeddings
fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() {
        return 0.0;
    }

    let dot_product: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();

    if norm_a == 0.0 || norm_b == 0.0 {
        return 0.0;
    }

    dot_product / (norm_a * norm_b)
}
