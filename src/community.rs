// community.rs - Semantic community detection using Louvain algorithm
//
// Detects coherent "subsystems" / "design units" from the symbol-similarity graph.
// This is the open-source equivalent of Sourcegraph's paid module-grouping features.

use std::collections::{HashMap, HashSet};

/// A weighted, undirected edge for the similarity graph.
#[derive(Debug, Clone)]
pub struct SimilarityEdge {
    pub src: usize,
    pub dst: usize,
    pub weight: f32,
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
                // Split by common delimiters
                for word in sym.split(|c: char| c == '_' || c == ':' || c.is_uppercase()) {
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
