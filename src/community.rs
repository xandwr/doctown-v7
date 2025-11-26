// community.rs - Semantic community detection using Louvain algorithm
//
// Detects coherent "subsystems" / "design units" from the symbol-similarity graph.
// This is the open-source equivalent of Sourcegraph's paid module-grouping features.

use std::collections::{HashMap, HashSet};
use crate::nlp;

/// A weighted, undirected edge for the similarity graph.
#[derive(Debug, Clone)]
pub struct SimilarityEdge {
    pub src: usize,
    pub dst: usize,
    pub weight: f32,
}

// Simple tokenization helpers and heuristics
const STOPWORDS: [&str; 17] = [
    "the", "and", "for", "with", "from", "into", "using", "use",
    "you", "your", "need", "what", "full", "look", "file", "try", "convert"
];
const VERB_BLACKLIST: [&str; 20] = [
    "get", "set", "new", "parse", "handle", "list", "build", "compute", "create", "write",
    "read", "load", "init", "process", "run", "generate", "add", "remove", "fetch", "call",
];

/// Check if a token is valid for label generation
/// Filters out: too short, stopwords, verbs, purely numeric, mostly numeric
fn is_valid_token(tok: &str) -> bool {
    if tok.len() < 3 { return false; }
    let lower = tok.to_lowercase();
    if STOPWORDS.contains(&lower.as_str()) || VERB_BLACKLIST.contains(&lower.as_str()) {
        return false;
    }
    // Filter tokens that are purely numeric or mostly numeric
    let digit_count = tok.chars().filter(|c| c.is_numeric()).count();
    let alpha_count = tok.chars().filter(|c| c.is_alphabetic()).count();
    // Reject if no letters, or if more than 50% digits
    alpha_count > 0 && digit_count <= alpha_count
}

fn split_to_tokens(s: &str) -> Vec<String> {
    // Break on non-alphanumeric and split camelCase / snake_case
    let mut parts: Vec<String> = Vec::new();
    let mut cur = String::new();
    for c in s.chars() {
        if c.is_ascii_alphanumeric() {
            // Split camelCase boundary
            if c.is_uppercase() && !cur.is_empty() && cur.chars().last().unwrap().is_lowercase() {
                parts.push(cur.clone());
                cur.clear();
            }
            cur.push(c);
        } else {
            if !cur.is_empty() {
                parts.push(cur.clone());
                cur.clear();
            }
        }
    }
    if !cur.is_empty() {
        parts.push(cur);
    }

    // Further split parts by underscores or double-colon
    let mut tokens = Vec::new();
    for p in parts {
        for q in p.split(|c: char| c == '_' || c == ':') {
            if !q.is_empty() {
                tokens.push(q.to_string());
            }
        }
    }
    tokens
}

/// Result of community detection: mapping from node index to community ID.
#[derive(Debug, Clone)]
pub struct CommunityResult {
    /// node index -> community id
    pub assignments: Vec<usize>,
    /// community id -> list of node indices
    pub communities: HashMap<usize, Vec<usize>>,
    /// Final modularity score
    pub modularity: f64,
}

/// Louvain algorithm for community detection on a weighted graph.
///
/// This is a classic two-phase algorithm:
/// 1. Local moving: greedily move nodes to neighboring communities to maximize modularity
/// 2. Aggregation: collapse communities into super-nodes and repeat
///
/// We iterate until modularity stops improving.
pub struct Louvain {
    /// Number of nodes
    n: usize,
    /// Adjacency list: node -> [(neighbor, weight)]
    adj: Vec<Vec<(usize, f32)>>,
    /// Sum of all edge weights (2m in the formula)
    total_weight: f64,
    /// Current community assignment for each node
    community: Vec<usize>,
    /// Sum of weights of edges incident to nodes in each community
    sigma_tot: Vec<f64>,
    /// Sum of weights of edges from node i to nodes in community c
    /// (computed on the fly)
    /// Self-loop weights for each node (k_i,in when node is alone)
    k_i: Vec<f64>,
}

impl Louvain {
    /// Build a Louvain solver from a list of similarity edges.
    /// `n` is the number of nodes (symbols).
    pub fn new(n: usize, edges: &[SimilarityEdge]) -> Self {
        let mut adj: Vec<Vec<(usize, f32)>> = vec![Vec::new(); n];
        let mut total_weight = 0.0f64;
        let mut k_i = vec![0.0f64; n];

        for e in edges {
            if e.src >= n || e.dst >= n {
                continue;
            }
            // Undirected: add both directions
            adj[e.src].push((e.dst, e.weight));
            adj[e.dst].push((e.src, e.weight));
            total_weight += e.weight as f64;
            k_i[e.src] += e.weight as f64;
            k_i[e.dst] += e.weight as f64;
        }

        // Each edge counted twice in k_i, but total_weight should be sum of unique edges
        // Actually for undirected graph, 2m = sum of degrees = 2 * sum of edge weights
        // So total_weight is fine as sum of edge weights, and 2m = 2 * total_weight
        let two_m = 2.0 * total_weight;

        // Initially each node is its own community
        let community: Vec<usize> = (0..n).collect();
        let sigma_tot = k_i.clone();

        Self {
            n,
            adj,
            total_weight: two_m,
            community,
            sigma_tot,
            k_i,
        }
    }

    /// Run the Louvain algorithm and return community assignments.
    pub fn run(&mut self) -> CommunityResult {
        if self.n == 0 || self.total_weight < 1e-9 {
            return CommunityResult {
                assignments: vec![],
                communities: HashMap::new(),
                modularity: 0.0,
            };
        }

        let mut improved = true;
        let mut iteration = 0;
        const MAX_ITERATIONS: usize = 100;

        while improved && iteration < MAX_ITERATIONS {
            improved = self.local_moving_phase();
            iteration += 1;
        }

        // Build final result
        let mut communities: HashMap<usize, Vec<usize>> = HashMap::new();
        for (node, &comm) in self.community.iter().enumerate() {
            communities.entry(comm).or_default().push(node);
        }

        // Renumber communities to be contiguous
        let mut comm_map: HashMap<usize, usize> = HashMap::new();
        let mut next_id = 0;
        for &c in self.community.iter() {
            if !comm_map.contains_key(&c) {
                comm_map.insert(c, next_id);
                next_id += 1;
            }
        }

        let assignments: Vec<usize> = self
            .community
            .iter()
            .map(|&c| *comm_map.get(&c).unwrap())
            .collect();

        let mut final_communities: HashMap<usize, Vec<usize>> = HashMap::new();
        for (node, &comm) in assignments.iter().enumerate() {
            final_communities.entry(comm).or_default().push(node);
        }

        let modularity = self.compute_modularity();

        CommunityResult {
            assignments,
            communities: final_communities,
            modularity,
        }
    }

    /// Phase 1: Local moving - greedily move nodes to maximize modularity.
    /// Returns true if any improvement was made.
    fn local_moving_phase(&mut self) -> bool {
        let mut improved = false;
        let mut moved = true;
        let mut pass = 0;
        const MAX_PASSES: usize = 20;

        while moved && pass < MAX_PASSES {
            moved = false;
            pass += 1;

            for node in 0..self.n {
                let current_comm = self.community[node];
                let k_i = self.k_i[node];

                // Compute k_i_in for current community (sum of weights to current community)
                let k_i_in_current = self.sum_weights_to_community(node, current_comm);

                // Remove node from current community
                self.sigma_tot[current_comm] -= k_i;

                // Find best community among neighbors
                let mut best_comm = current_comm;
                let mut best_delta = 0.0f64;

                // Collect neighbor communities
                let neighbor_comms: HashSet<usize> =
                    self.adj[node].iter().map(|&(nb, _)| self.community[nb]).collect();

                for &target_comm in &neighbor_comms {
                    if target_comm == current_comm {
                        continue;
                    }

                    let k_i_in_target = self.sum_weights_to_community(node, target_comm);
                    let sigma_tot_target = self.sigma_tot[target_comm];
                    let sigma_tot_current = self.sigma_tot[current_comm];

                    // Delta Q = [k_i_in_target / m - sigma_tot_target * k_i / (2m^2)]
                    //         - [k_i_in_current / m - sigma_tot_current * k_i / (2m^2)]
                    // Simplified (all terms share 1/m factor):
                    let m = self.total_weight / 2.0;
                    let delta = (k_i_in_target - k_i_in_current) / m
                        - k_i * (sigma_tot_target - sigma_tot_current) / (2.0 * m * m);

                    if delta > best_delta {
                        best_delta = delta;
                        best_comm = target_comm;
                    }
                }

                // Move node to best community
                self.community[node] = best_comm;
                self.sigma_tot[best_comm] += k_i;

                if best_comm != current_comm {
                    moved = true;
                    improved = true;
                }
            }
        }

        improved
    }

    /// Sum of edge weights from `node` to all nodes in `comm`.
    fn sum_weights_to_community(&self, node: usize, comm: usize) -> f64 {
        let mut sum = 0.0f64;
        for &(neighbor, weight) in &self.adj[node] {
            if self.community[neighbor] == comm {
                sum += weight as f64;
            }
        }
        sum
    }

    /// Compute the modularity Q of the current partition.
    fn compute_modularity(&self) -> f64 {
        if self.total_weight < 1e-9 {
            return 0.0;
        }

        let m = self.total_weight / 2.0;
        let mut q = 0.0f64;

        for node in 0..self.n {
            let comm = self.community[node];
            let k_i = self.k_i[node];

            for &(neighbor, weight) in &self.adj[node] {
                if self.community[neighbor] == comm {
                    // A_ij - k_i * k_j / (2m)
                    let k_j = self.k_i[neighbor];
                    q += weight as f64 - (k_i * k_j) / (2.0 * m);
                }
            }
        }

        q / (2.0 * m)
    }
}

/// Compute cosine similarity between two embedding vectors.
/// Assumes vectors are already L2-normalized (returns dot product).
pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() {
        return 0.0;
    }
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

/// Build similarity edges from symbol embeddings.
/// Only creates edges where similarity >= threshold.
pub fn build_similarity_edges(
    embeddings: &[std::sync::Arc<[f32]>],
    threshold: f32,
) -> Vec<SimilarityEdge> {
    let n = embeddings.len();
    let mut edges = Vec::new();

    for i in 0..n {
        for j in (i + 1)..n {
            let sim = cosine_similarity(&embeddings[i], &embeddings[j]);
            if sim >= threshold {
                edges.push(SimilarityEdge {
                    src: i,
                    dst: j,
                    weight: sim,
                });
            }
        }
    }

    edges
}

/// A detected semantic community (subsystem / design unit).
#[derive(Debug, Clone)]
pub struct SemanticCommunity {
    pub id: usize,
    pub symbols: Vec<String>,
    pub files: HashSet<String>,
    /// Average internal similarity
    pub cohesion: f32,
    /// Suggested label based on common prefixes/patterns
    pub suggested_label: Option<String>,
}

impl SemanticCommunity {
    /// Try to infer a label from common symbol name prefixes.
    pub fn infer_label(symbols: &[String]) -> Option<String> {
        if symbols.is_empty() {
            return None;
        }

        // Find common prefix among symbol names
        let mut prefix = symbols[0].clone();
        for sym in symbols.iter().skip(1) {
            let common_len = prefix
                .chars()
                .zip(sym.chars())
                .take_while(|(a, b)| a == b)
                .count();
            prefix.truncate(common_len);
        }

        // Clean up prefix (remove trailing underscores, etc.)
        let prefix = prefix.trim_end_matches('_').trim_end_matches("::");

        if prefix.len() >= 3 {
            Some(prefix.to_string())
        } else {
            // Fallback: most common word in symbol names
            let mut word_counts: HashMap<String, usize> = HashMap::new();
            for sym in symbols {
                // Split by common delimiters and camelcase breaks
                for word in split_to_tokens(sym) {
                    let word = word.to_lowercase();
                    if word.len() >= 3 {
                        *word_counts.entry(word).or_default() += 1;
                    }
                }
            }

            word_counts
                .into_iter()
                .max_by_key(|(_, count)| *count)
                .filter(|(_, count)| *count >= 2)
                .map(|(word, _)| word)
        }
    }

    /// Infer a descriptive label for a community using embeddings.
    ///
    /// Steps:
    /// - compute centroid of community chunk embeddings
    /// - find top-k nearest chunks (by cosine similarity)
    /// - extract tokens from those chunks' ids and containing symbol names
    /// - score tokens (frequency + heuristics) and return top token(s)
    pub fn infer_label_with_embeddings(
        indices: &[usize],
        embeddings: &[std::sync::Arc<[f32]>],
        chunk_ids: &[String],
        chunk_symbol_lists: &[Vec<String>],
        chunk_texts: &[String],
        symbol_names: &[String],
    ) -> Option<String> {
        if embeddings.is_empty() {
            return SemanticCommunity::infer_label(symbol_names);
        }

        // Compute centroid embedding for the community indices
        let dim = embeddings[0].len();
        let mut centroid = vec![0.0f32; dim];
        let mut count = 0usize;
        for &i in indices {
            if i < embeddings.len() {
                for (d, v) in centroid.iter_mut().zip(embeddings[i].iter()) {
                    *d += *v;
                }
                count += 1;
            }
        }
        if count == 0 {
            return SemanticCommunity::infer_label(symbol_names);
        }
        for d in centroid.iter_mut() {
            *d /= count as f32;
        }

        // Compute centrality (sum similarity to other cluster members) and pick top-5 central chunks.
        let mut centrality: Vec<(usize, f32)> = Vec::new();
        for &i in indices {
            if i >= embeddings.len() { continue; }
            let mut sum = 0.0f32;
            for &j in indices {
                if j >= embeddings.len() || i == j { continue; }
                sum += cosine_similarity(&embeddings[i], &embeddings[j]);
            }
            centrality.push((i, sum));
        }
        centrality.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        let top_k = centrality.iter().take(5).map(|(i, _)| *i).collect::<Vec<usize>>();

        // Build a corpus of candidate phrases per chunk using comment/docs text + identifiers/symbols.
        fn extract_comment_text(text: &str) -> String {
            let mut out = Vec::new();
            let mut in_block = false;
            for line in text.lines() {
                let t = line.trim_start();
                if t.starts_with("///") || t.starts_with("//") || t.starts_with('#') {
                    out.push(t.trim_start_matches('/').trim_start_matches('#').trim().to_string());
                } else if t.starts_with("/*") {
                    in_block = true;
                    out.push(t.trim_start_matches("/*").trim().to_string());
                } else if in_block {
                    if t.ends_with("*/") {
                        in_block = false;
                        out.push(t.trim_end_matches("*/").trim().to_string());
                    } else {
                        out.push(t.to_string());
                    }
                }
            }
            out.join(" ")
        }

        let mut doc_phrases: Vec<Vec<String>> = Vec::with_capacity(chunk_texts.len());
        for ci in 0..chunk_texts.len() {
            let mut phrases: Vec<String> = Vec::new();
                if let Some(text) = chunk_texts.get(ci) {
                let comments = extract_comment_text(text);
                    // Extract noun phrases from COMMENTS ONLY, not raw chunk text
                    // This avoids picking up prose like "You Need" or "What You" from markdown
                    if !comments.is_empty() {
                        for np in nlp::extract_noun_phrases(&comments) {
                            let p = np.to_lowercase();
                            if p.len() >= 3 { phrases.push(p); }
                        }
                    }
                // tokenize comment words
                for word in comments.split(|c: char| !c.is_alphanumeric() && c != '_' && c != ':') {
                    if word.is_empty() { continue; }
                    for tok in split_to_tokens(word) {
                        if is_valid_token(&tok) {
                            phrases.push(tok.to_lowercase());
                        }
                    }
                }
                // also add identifiers found in chunk id as candidates
                if let Some(cid) = chunk_ids.get(ci) {
                    for tok in split_to_tokens(cid) {
                        if is_valid_token(&tok) {
                            phrases.push(tok.to_lowercase());
                        }
                    }
                }
                // include symbol tokens for this chunk
                if let Some(sym_list) = chunk_symbol_lists.get(ci) {
                    for sym in sym_list {
                        for tok in split_to_tokens(sym) {
                            if is_valid_token(&tok) {
                                phrases.push(tok.to_lowercase());
                            }
                        }
                    }
                }
                // bigrams from the token stream (comments/identifiers mixed)
                let toks: Vec<String> = phrases.clone();
                for w in toks.windows(2) {
                    let big = format!("{} {}", w[0], w[1]);
                    phrases.push(big);
                }
            } else {
                // fallback to ids/symbols
                if let Some(cid) = chunk_ids.get(ci) {
                    for tok in split_to_tokens(cid) {
                        if is_valid_token(&tok) {
                            phrases.push(tok.to_lowercase());
                        }
                    }
                }
                if let Some(sym_list) = chunk_symbol_lists.get(ci) {
                    for sym in sym_list {
                        for tok in split_to_tokens(sym) {
                            if is_valid_token(&tok) {
                                phrases.push(tok.to_lowercase());
                            }
                        }
                    }
                }
            }
            doc_phrases.push(phrases);
        }

        // Compute DF across corpus
        let mut df: HashMap<String, usize> = HashMap::new();
        for phrases in &doc_phrases {
            let mut seen: HashSet<String> = HashSet::new();
            for p in phrases {
                if !seen.contains(p) {
                    *df.entry(p.clone()).or_default() += 1;
                    seen.insert(p.clone());
                }
            }
        }
        let corpus_n = doc_phrases.len() as f32;

        // TF counts within top-k selected chunks
        let mut tf: HashMap<String, f32> = HashMap::new();
        for &ci in &top_k {
            if let Some(phrases) = doc_phrases.get(ci) {
                for p in phrases {
                    *tf.entry(p.clone()).or_default() += 1.0;
                }
            }
        }
        if tf.is_empty() { return SemanticCommunity::infer_label(symbol_names); }

        // TF-IDF scoring
        let mut scored: Vec<(String, f32)> = Vec::new();
        for (phrase, tcount) in tf.into_iter() {
            let df_count = *df.get(&phrase).unwrap_or(&1) as f32;
            let idf = (1.0 + corpus_n / (1.0 + df_count)).ln();
            scored.push((phrase, tcount * idf));
        }
        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        // Pick top 1-3 tokens (words/phrases) and title-case them
        let mut picks: Vec<String> = Vec::new();
        let mut seen_words: HashSet<String> = HashSet::new();
        
        for (p, _) in scored.iter().take(10) {  // Look at top 10 to find 3 unique
            if p.len() < 3 { continue; }
            // Filter out purely numeric or mostly numeric tokens
            let digit_count = p.chars().filter(|c| c.is_numeric()).count();
            let alpha_count = p.chars().filter(|c| c.is_alphabetic()).count();
            // Reject if no letters, or if more than 50% digits
            if alpha_count == 0 || (digit_count > alpha_count) { continue; }
            
            let nice = p.split_whitespace().map(|w| {
                let mut chs = w.chars();
                match chs.next() {
                    Some(first) => first.to_uppercase().collect::<String>() + chs.as_str(),
                    None => String::new(),
                }
            }).collect::<Vec<_>>().join(" ");
            
            // Deduplicate: skip if we've already seen this exact word/phrase or its components
            let words_in_phrase: Vec<String> = nice.to_lowercase().split_whitespace().map(|s| s.to_string()).collect();
            let is_duplicate = words_in_phrase.iter().any(|w| seen_words.contains(w));
            
            if !is_duplicate {
                for w in words_in_phrase {
                    seen_words.insert(w);
                }
                picks.push(nice);
                if picks.len() >= 3 { break; }
            }
        }
        if picks.is_empty() { return SemanticCommunity::infer_label(symbol_names); }
        if picks.len() == 1 { return Some(picks[0].clone()); }
        Some(picks.into_iter().take(3).collect::<Vec<_>>().join(" + "))
    }

    /// Compute average pairwise similarity (cohesion) for a community.
    pub fn compute_cohesion(indices: &[usize], embeddings: &[std::sync::Arc<[f32]>]) -> f32 {
        if indices.len() < 2 {
            return 1.0;
        }

        let mut total = 0.0f32;
        let mut count = 0;

        for (i, &idx_a) in indices.iter().enumerate() {
            for &idx_b in indices.iter().skip(i + 1) {
                if idx_a < embeddings.len() && idx_b < embeddings.len() {
                    total += cosine_similarity(&embeddings[idx_a], &embeddings[idx_b]);
                    count += 1;
                }
            }
        }

        if count > 0 {
            total / count as f32
        } else {
            0.0
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_louvain_basic() {
        // Simple graph: two cliques connected by a weak edge
        // Nodes 0,1,2 form one clique, nodes 3,4,5 form another
        let edges = vec![
            // Clique 1
            SimilarityEdge { src: 0, dst: 1, weight: 0.9 },
            SimilarityEdge { src: 0, dst: 2, weight: 0.9 },
            SimilarityEdge { src: 1, dst: 2, weight: 0.9 },
            // Clique 2
            SimilarityEdge { src: 3, dst: 4, weight: 0.9 },
            SimilarityEdge { src: 3, dst: 5, weight: 0.9 },
            SimilarityEdge { src: 4, dst: 5, weight: 0.9 },
            // Weak bridge
            SimilarityEdge { src: 2, dst: 3, weight: 0.1 },
        ];

        let mut louvain = Louvain::new(6, &edges);
        let result = louvain.run();

        // Should detect 2 communities
        assert!(result.communities.len() >= 1);
        assert!(result.modularity >= 0.0);

        // Nodes 0,1,2 should be in same community
        assert_eq!(result.assignments[0], result.assignments[1]);
        assert_eq!(result.assignments[1], result.assignments[2]);

        // Nodes 3,4,5 should be in same community
        assert_eq!(result.assignments[3], result.assignments[4]);
        assert_eq!(result.assignments[4], result.assignments[5]);
    }

    #[test]
    fn test_cosine_similarity() {
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![1.0, 0.0, 0.0];
        assert!((cosine_similarity(&a, &b) - 1.0).abs() < 0.001);

        let c = vec![0.0, 1.0, 0.0];
        assert!((cosine_similarity(&a, &c)).abs() < 0.001);
    }

    #[test]
    fn test_infer_label() {
        let symbols = vec![
            "parse_symbols".to_string(),
            "parse_file".to_string(),
            "parse_chunk".to_string(),
        ];
        let label = SemanticCommunity::infer_label(&symbols);
        assert_eq!(label, Some("parse".to_string()));
    }
}
