use anyhow::Result;
use ort::session::Session;
use ort::value::Value;
use std::sync::Arc;
use tokenizers::Tokenizer;

pub struct EmbeddingEngine {
    tokenizer: Tokenizer,
    session: Session,
}

impl EmbeddingEngine {
    pub fn new(model_path: &str, tokenizer_path: &str) -> Result<Self> {
        // Create session with CUDA support (falls back to CPU automatically)
        let session = Session::builder()?.commit_from_file(model_path)?;

        let tokenizer = Tokenizer::from_file(tokenizer_path)
            .map_err(|e| anyhow::anyhow!("Failed to load tokenizer: {}", e))?;

        Ok(Self { tokenizer, session })
    }

    pub fn embed(&mut self, text: &str) -> Result<Arc<[f32]>> {
        // 1) tokenize
        let encoding = self
            .tokenizer
            .encode(text, true)
            .map_err(|e| anyhow::anyhow!("Failed to encode text: {}", e))?;
        
        // Clamp token IDs to valid range for the model (BERT vocab size is typically 30522)
        // Token IDs outside this range get mapped to [UNK] token (100)
        const MAX_TOKEN_ID: u32 = 30521;
        const UNK_TOKEN_ID: i64 = 100;
        
        let input_ids = encoding
            .get_ids()
            .iter()
            .map(|&id| {
                if id > MAX_TOKEN_ID {
                    UNK_TOKEN_ID
                } else {
                    id as i64
                }
            })
            .collect::<Vec<_>>();

        // 2) create input tensors
        let len = input_ids.len();
        let shape = vec![1, len];
        
        // Create input_ids tensor
        let input_tensor = Value::from_array((shape.as_slice(), input_ids))?;
        
        // Create attention_mask tensor (all ones)
        let attention_mask: Vec<i64> = vec![1; len];
        let attention_mask_tensor = Value::from_array((shape.as_slice(), attention_mask))?;
        
        // Create token_type_ids tensor (all zeros for single segment)
        let token_type_ids: Vec<i64> = vec![0; len];
        let token_type_ids_tensor = Value::from_array((shape.as_slice(), token_type_ids))?;

        // 3) run session with all three inputs
        let outputs = self.session.run(ort::inputs![
            "input_ids" => input_tensor,
            "attention_mask" => attention_mask_tensor,
            "token_type_ids" => token_type_ids_tensor
        ])?;

        // 4) extract embedding - get the first output and convert to Vec
        let output_tensor = outputs[0].try_extract_tensor::<f32>()?;
        let (_shape, data) = output_tensor;
        let mut embedding = data.to_vec();

        // L2-normalize embedding so cosine similarity works reliably
        Self::l2_normalize(&mut embedding);

        // Convert Vec<f32> -> Arc<[f32]> for cheap clone/share semantics
        let arc_embedding: Arc<[f32]> = Arc::from(embedding.into_boxed_slice());
        Ok(arc_embedding)
    }

    pub fn embed_batch(&mut self, inputs: &[String]) -> Result<Vec<Arc<[f32]>>> {
        let mut embeddings = Vec::with_capacity(inputs.len());
        for text in inputs {
            embeddings.push(self.embed(text)?);
        }
        Ok(embeddings)
    }

    // Normalize vector in-place to unit length (L2). If vector has near-zero
    // norm it is left unchanged to avoid dividing by zero.
    fn l2_normalize(vec: &mut [f32]) {
        let sum_sq: f32 = vec.iter().map(|&v| v * v).sum();
        let norm = sum_sq.sqrt();
        if norm > 1e-12 {
            for v in vec.iter_mut() {
                *v /= norm;
            }
        }
    }
}
