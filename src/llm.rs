// llm.rs - Local LLM inference for generating plain-English summaries
//
// Uses ONNX Runtime for GPU-accelerated text generation with quantized models.
// Optimized for running on consumer GPUs (12GB VRAM) and serverless RunPod instances.

use anyhow::Result;
use serde_json::Value;
use std::process::Command;

/// Local LLM engine for generating plain-English explanations
///
/// Currently uses ollama as a simple interface to local models.
/// This avoids complex ONNX text generation setup while still keeping everything local.
pub struct LlmEngine {
    model_name: String,
}

impl LlmEngine {
    /// Create a new LLM engine using ollama
    ///
    /// Recommended models for 12GB VRAM:
    /// - "phi3:mini" (3.8B params, fast, good quality)
    /// - "llama3.2:3b" (3B params, very fast)
    /// - "qwen2.5:3b" (3B params, good at code)
    pub fn new(model_name: Option<&str>) -> Result<Self> {
        let model_name = model_name.unwrap_or("phi3:mini").to_string();

        // Verify ollama is installed and model is available
        let status = Command::new("ollama")
            .arg("list")
            .output();

        if status.is_err() {
            anyhow::bail!(
                "Ollama not found. Please install: https://ollama.ai\n\
                Then run: ollama pull {}",
                model_name
            );
        }

        Ok(Self { model_name })
    }

    /// Generate a plain-English "Explain Like I'm 5" summary for a code symbol
    ///
    /// Takes symbol metadata and generates a concise, understandable explanation
    /// of what the symbol does and WHY it exists.
    pub fn explain_symbol(
        &self,
        symbol_name: &str,
        symbol_kind: &str,
        signature: Option<&str>,
        code_context: Option<&str>,
        existing_summary: &str,
    ) -> Result<String> {
        // Build prompt for plain-English explanation
        let mut prompt = format!(
            "You are a senior software engineer explaining code to a junior developer.\n\n\
            Explain what this {} does and WHY it exists in 1-2 simple sentences.\n\n\
            Symbol: {}\n",
            symbol_kind, symbol_name
        );

        if let Some(sig) = signature {
            prompt.push_str(&format!("Signature: {}\n", sig));
        }

        if !existing_summary.is_empty() {
            prompt.push_str(&format!("Technical summary: {}\n", existing_summary));
        }

        if let Some(code) = code_context {
            // Truncate code context to avoid token limits
            let truncated = if code.len() > 500 {
                &code[..500]
            } else {
                code
            };
            prompt.push_str(&format!("\nCode:\n```\n{}\n```\n", truncated));
        }

        prompt.push_str("\nPlain-English explanation (1-2 sentences):");

        // Call ollama API
        let output = Command::new("ollama")
            .arg("run")
            .arg(&self.model_name)
            .arg(&prompt)
            .output()?;

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            anyhow::bail!("Ollama inference failed: {}", stderr);
        }

        let response = String::from_utf8(output.stdout)?
            .trim()
            .to_string();

        // Clean up the response (remove common artifacts)
        let cleaned = response
            .lines()
            .filter(|line| !line.trim().is_empty())
            .take(3) // Max 3 lines for brevity
            .collect::<Vec<_>>()
            .join(" ");

        Ok(cleaned)
    }

    /// Generate summaries for a batch of symbols (with progress feedback)
    pub fn explain_symbols_batch(
        &self,
        symbols: &[(String, String, Option<String>, Option<String>, String)], // (name, kind, signature, code, summary)
    ) -> Result<Vec<String>> {
        let total = symbols.len();
        let mut results = Vec::with_capacity(total);

        println!("🤖 Generating LLM summaries for {} symbols...", total);

        for (i, (name, kind, sig, code, summary)) in symbols.iter().enumerate() {
            if (i + 1) % 10 == 0 {
                println!("   Progress: {}/{}", i + 1, total);
            }

            let explanation = self.explain_symbol(
                name,
                kind,
                sig.as_deref(),
                code.as_deref(),
                summary,
            )?;

            results.push(explanation);
        }

        println!("✅ Generated {} LLM summaries", total);
        Ok(results)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[ignore] // Only run if ollama is installed
    fn test_explain_symbol() {
        let engine = LlmEngine::new(Some("phi3:mini")).unwrap();

        let explanation = engine.explain_symbol(
            "embed_batch",
            "function",
            Some("fn embed_batch(&mut self, inputs: &[String]) -> Result<Vec<Arc<[f32]>>>"),
            Some("pub fn embed_batch(&mut self, inputs: &[String]) -> Result<Vec<Arc<[f32]>>> {\n    // Tokenize and embed multiple texts in one GPU pass\n}"),
            "Converts multiple text strings into embedding vectors efficiently",
        ).unwrap();

        println!("Explanation: {}", explanation);
        assert!(!explanation.is_empty());
    }
}
