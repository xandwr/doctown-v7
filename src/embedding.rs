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

    #[allow(dead_code)]
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
        let (shape, data) = output_tensor;
        
        // The output shape is [batch_size, seq_len, hidden_dim]
        // For sentence embeddings, we need to do mean pooling over seq_len
        // Shape: [1, seq_len, 384] -> [384]
        let batch_size = shape[0];
        let seq_len = shape[1] as usize;
        let hidden_dim = shape[2] as usize;
        
        if batch_size != 1 {
            anyhow::bail!("Expected batch_size=1, got {}", batch_size);
        }
        
        // Mean pooling: average the embeddings across the sequence length
        let mut embedding = vec![0.0f32; hidden_dim];
        for i in 0..seq_len {
            for j in 0..hidden_dim {
                let idx = i * hidden_dim + j;
                embedding[j] += data[idx];
            }
        }
        
        // Divide by sequence length to get the mean
        for val in &mut embedding {
            *val /= seq_len as f32;
        }

        // L2-normalize embedding so cosine similarity works reliably
        Self::l2_normalize(&mut embedding);

        // Convert Vec<f32> -> Arc<[f32]> for cheap clone/share semantics
        let arc_embedding: Arc<[f32]> = Arc::from(embedding.into_boxed_slice());
        Ok(arc_embedding)
    }

    pub fn embed_batch(&mut self, inputs: &[String]) -> Result<Vec<Arc<[f32]>>> {
        if inputs.is_empty() {
            return Ok(Vec::new());
        }

        // Tokenize all inputs
        let mut all_input_ids = Vec::with_capacity(inputs.len());
        let mut all_attention_masks = Vec::with_capacity(inputs.len());
        let mut all_token_type_ids = Vec::with_capacity(inputs.len());
        let mut max_len = 0;

        for text in inputs {
            let encoding = self
                .tokenizer
                .encode(text.as_str(), true)
                .map_err(|e| anyhow::anyhow!("Failed to encode text: {}", e))?;

            // Clamp token IDs to valid range
            const MAX_TOKEN_ID: u32 = 30521;
            const UNK_TOKEN_ID: i64 = 100;

            let input_ids: Vec<i64> = encoding
                .get_ids()
                .iter()
                .map(|&id| {
                    if id > MAX_TOKEN_ID {
                        UNK_TOKEN_ID
                    } else {
                        id as i64
                    }
                })
                .collect();

            let len = input_ids.len();
            max_len = max_len.max(len);

            all_input_ids.push(input_ids);
            all_attention_masks.push(vec![1i64; len]);
            all_token_type_ids.push(vec![0i64; len]);
        }

        // Pad all sequences to max_len
        for i in 0..inputs.len() {
            let current_len = all_input_ids[i].len();
            if current_len < max_len {
                let pad_len = max_len - current_len;
                all_input_ids[i].extend(vec![0i64; pad_len]);
                all_attention_masks[i].extend(vec![0i64; pad_len]);
                all_token_type_ids[i].extend(vec![0i64; pad_len]);
            }
        }

        // Flatten into batch tensors [batch_size, max_len]
        let batch_size = inputs.len();
        let shape = vec![batch_size, max_len];

        let flat_input_ids: Vec<i64> = all_input_ids.into_iter().flatten().collect();
        let flat_attention_masks: Vec<i64> = all_attention_masks.into_iter().flatten().collect();
        let flat_token_type_ids: Vec<i64> = all_token_type_ids.into_iter().flatten().collect();

        // Create batch tensors
        let input_tensor = Value::from_array((shape.as_slice(), flat_input_ids))?;
        let attention_mask_tensor = Value::from_array((shape.as_slice(), flat_attention_masks))?;
        let token_type_ids_tensor = Value::from_array((shape.as_slice(), flat_token_type_ids))?;

        // Run batch inference
        let outputs = self.session.run(ort::inputs![
            "input_ids" => input_tensor,
            "attention_mask" => attention_mask_tensor,
            "token_type_ids" => token_type_ids_tensor
        ])?;

        // Extract embeddings - output shape is [batch_size, embedding_dim]
        let output_tensor = outputs[0].try_extract_tensor::<f32>()?;
        let (output_shape, data) = output_tensor;

        // Determine embedding dimension from output shape
        let embedding_dim = if output_shape.len() >= 2 {
            output_shape[output_shape.len() - 1] as usize
        } else {
            return Err(anyhow::anyhow!(
                "Unexpected output shape: {:?}",
                output_shape
            ));
        };

        // Split batch results into individual embeddings
        let mut embeddings = Vec::with_capacity(batch_size);
        for i in 0..batch_size {
            let start = i * embedding_dim;
            let end = start + embedding_dim;
            let mut embedding = data[start..end].to_vec();

            // L2-normalize each embedding
            Self::l2_normalize(&mut embedding);

            let arc_embedding: Arc<[f32]> = Arc::from(embedding.into_boxed_slice());
            embeddings.push(arc_embedding);
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
