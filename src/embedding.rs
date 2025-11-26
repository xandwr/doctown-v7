use anyhow::Result;
use ort::session::Session;
use ort::value::Value;
use tokenizers::Tokenizer;

pub struct EmbeddingEngine {
    tokenizer: Tokenizer,
    session: Session,
}

impl EmbeddingEngine {
    pub fn new(model_path: &str, tokenizer_path: &str) -> Result<Self> {
        // Create session with CUDA support (falls back to CPU automatically)
        let session = Session::builder()?
            .commit_from_file(model_path)?;

        let tokenizer = Tokenizer::from_file(tokenizer_path)
            .map_err(|e| anyhow::anyhow!("Failed to load tokenizer: {}", e))?;

        Ok(Self { tokenizer, session })
    }

    pub fn embed(&mut self, text: &str) -> Result<Vec<f32>> {
        // 1) tokenize
        let encoding = self.tokenizer.encode(text, true)
            .map_err(|e| anyhow::anyhow!("Failed to encode text: {}", e))?;
        let input_ids = encoding.get_ids().iter().map(|&id| id as i64).collect::<Vec<_>>();

        // 2) create input tensor
        let len = input_ids.len();
        let shape = vec![1, len];
        let input_tensor = Value::from_array((shape.as_slice(), input_ids))?;

        // 3) run session
        let outputs = self.session.run(ort::inputs![input_tensor])?;

        // 4) extract embedding - get the first output and convert to Vec
        let output_tensor = outputs[0].try_extract_tensor::<f32>()?;
        let (_shape, data) = output_tensor;
        let embedding = data.to_vec();
        
        Ok(embedding)
    }

    pub fn embed_batch(&mut self, inputs: &[String]) -> Result<Vec<Vec<f32>>> {
        let mut embeddings = Vec::new();
        for text in inputs {
            embeddings.push(self.embed(text)?);
        }
        Ok(embeddings)
    }
}
