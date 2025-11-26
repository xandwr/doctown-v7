// main.rs

mod ingest;
use anyhow::Result;
use ingest::load_zip;

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

    println!("{}:\t\t{:?}", source, &zip_bytes[zip_bytes.len().saturating_sub(10)..]);

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
