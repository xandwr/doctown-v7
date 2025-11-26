// nlp.rs - Rule-based noun phrase extraction with proper POS patterns

#[cfg(feature = "pos")]
pub fn extract_noun_phrases(text: &str) -> Vec<String> {
    use regex::Regex;
    use std::collections::HashSet;

    // Common determiners, prepositions, and other non-noun words to filter
    let stop_words: HashSet<&str> = [
        "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
        "have", "has", "had", "do", "does", "did", "will", "would", "should",
        "could", "may", "might", "must", "can", "to", "of", "in", "for", "on",
        "at", "by", "with", "from", "as", "into", "through", "during", "before",
        "after", "above", "below", "between", "under", "again", "further", "then",
        "once", "here", "there", "when", "where", "why", "how", "all", "both",
        "each", "few", "more", "most", "other", "some", "such", "no", "nor",
        "not", "only", "own", "same", "so", "than", "too", "very", "s", "t",
        "just", "don", "now", "d", "ll", "m", "o", "re", "ve", "y", "ain",
        "aren", "couldn", "didn", "doesn", "hadn", "hasn", "haven", "isn",
        "ma", "mightn", "mustn", "needn", "shan", "shouldn", "wasn", "weren",
        "won", "wouldn", "what", "you", "need", "finding", "full", "show",
    ]
    .iter()
    .cloned()
    .collect();

    // Tokenize into words (keeping alphanumeric and underscores)
    let word_regex = Regex::new(r"\b[A-Za-z_][A-Za-z0-9_]*\b").unwrap();
    let tokens: Vec<&str> = word_regex
        .find_iter(text)
        .map(|m| m.as_str())
        .collect();

    let mut noun_phrases = Vec::new();
    let mut current_phrase = Vec::new();

    for token in tokens {
        let lower = token.to_lowercase();
        
        // Check if it's a stop word or too short
        if stop_words.contains(lower.as_str()) || token.len() < 2 {
            if !current_phrase.is_empty() {
                noun_phrases.push(current_phrase.join(" "));
                current_phrase.clear();
            }
            continue;
        }

        // Skip pure numbers - they're not meaningful noun phrases
        if token.chars().all(|c| c.is_numeric()) {
            if !current_phrase.is_empty() {
                noun_phrases.push(current_phrase.join(" "));
                current_phrase.clear();
            }
            continue;
        }

        // Check if it looks like a noun (starts with capital or is CamelCase/PascalCase)
        let is_capitalized = token.chars().next().unwrap().is_uppercase();
        let has_internal_caps = token.chars().skip(1).any(|c| c.is_uppercase());
        let is_likely_noun = is_capitalized || has_internal_caps;

        // Also accept technical terms (snake_case, meaningful alphanumeric like V1, Http2)
        let has_underscore = token.contains('_');
        let has_digit = token.chars().any(|c| c.is_numeric());
        let has_alpha = token.chars().any(|c| c.is_alphabetic());
        // Only accept mixed alphanumeric if there are both letters and numbers
        let is_technical_term = has_underscore || (has_digit && has_alpha && token.len() > 2);

        if is_likely_noun || is_technical_term {
            current_phrase.push(token);
        } else {
            if !current_phrase.is_empty() {
                noun_phrases.push(current_phrase.join(" "));
                current_phrase.clear();
            }
        }
    }

    // Don't forget the last phrase
    if !current_phrase.is_empty() {
        noun_phrases.push(current_phrase.join(" "));
    }

    // Filter out single-character "phrases" and deduplicate
    let mut seen = HashSet::new();
    noun_phrases
        .into_iter()
        .filter(|p| p.len() > 1)
        .filter(|p| seen.insert(p.clone()))
        .collect()
}

#[cfg(not(feature = "pos"))]
pub fn extract_noun_phrases(_text: &str) -> Vec<String> {
    // Feature disabled: return empty so caller falls back to heuristics
    Vec::new()
}
