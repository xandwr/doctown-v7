// ingest.rs
// takes a path to a .zip file
// either locally or from a web url (url ends with .zip and it's a hosted file)

// goal: `doctown-builder https://github.com/user/repo/archive/refs/heads/main.zip`
// or: `cargo run -- https://github.com/xandwr/localdoc/archive/refs/heads/main.zip`

use anyhow::{Result, anyhow};
use reqwest::Client;
use tokio::fs;
use url::Url;

pub async fn load_zip(source: &str) -> Result<Vec<u8>> {
    // Basic guardrail
    if !source.ends_with(".zip") {
        return Err(anyhow!("Source does not end with .zip: {}", source));
    }

    // Try to parse as URL
    if let Ok(url) = Url::parse(source) {
        return load_zip_from_url(url).await;
    }

    // Otherwise treat as local path
    load_zip_from_path(source).await
}

async fn load_zip_from_path(path: &str) -> Result<Vec<u8>> {
    let data = fs::read(path).await?;
    Ok(data)
}

async fn load_zip_from_url(url: Url) -> Result<Vec<u8>> {
    let client = Client::new();
    let res = client.get(url.clone()).send().await?.error_for_status()?;

    let bytes = res.bytes().await?;
    Ok(bytes.to_vec())
}
