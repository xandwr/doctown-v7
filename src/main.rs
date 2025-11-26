// main.rs

mod ingest;
use anyhow::Result;
use ingest::{load_zip, unzip_to_memory, code_file_stats, FileKind};

#[tokio::main]
async fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();

    if args.len() < 2 {
        eprintln!("Usage: doctown-builder <zip-path-or-url>");
        std::process::exit(1);
    }

    let source = &args[1];

    // 1: Ingest
    let zip_bytes = load_zip(source).await?;
    let processed = unzip_to_memory(&zip_bytes).await?;

    println!("Processed {} files:", processed.len());
    for pf in &processed {
        // Get filename and kind from metadata
        let filename = pf.metadata.get("path").cloned()
            .or_else(|| pf.metadata.get("filename").cloned())
            .unwrap_or_default();
        let filetype = pf.metadata.get("filetype").cloned()
            .or_else(|| pf.metadata.get("language").cloned())
            .unwrap_or_default();

        // Filter out entries with missing/unknown filename or filetype
        if filename.is_empty() || filetype.is_empty() || filename == "<unknown>" || filetype == "<unknown>" {
            continue;
        }

        // Print filename and file kind
        print!(" - {} [{}]", filename, filetype);

        // Print detailed code stats for code files
        if ["rs", "py", "js", "ts", "cpp", "java"].contains(&filetype.as_str()) {
            if let Some(stats) = code_file_stats(&pf.original_bytes, &filetype) {
                let (total, code, comment, blank) = stats;
                print!(" | lines={} code={} comment={} blank={}", total, code, comment, blank);
            }
        }

        // Always print chunk/symbol/embed/meta counts
        println!(
            " | chunks={} symbols={} embeds={} metadata_keys={}",
            pf.chunks.len(),
            pf.symbols.len(),
            pf.embeddings.len(),
            pf.metadata.len()
        );
    }

    // Pass bytes into the pipeline later
    // ingest::run(zip_bytes).await?;

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_localdoc_repo() {
        let result =
            load_zip("https://github.com/xandwr/localdoc/archive/refs/heads/main.zip").await;
        assert!(result.is_ok());
    }
}
