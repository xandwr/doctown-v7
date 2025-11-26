
2. Add search_by_signature
// search_symbols(query="fn new() -> Result")
// search_symbols(query="impl Iterator for")
This would be HUGE. I often know the shape of what I'm looking for but not the name.

3. "Hot Spots" Analysis
{
  "most_modified": [...],  // from git history
  "most_connected": [...], // from graph
  "complexity_bombs": [...], // high cyclomatic complexity
  "orphans": [...]  // low connectivity, might be dead code
}
This tells me "here's where the action is" or "here's the sketchy stuff."

6. "Explain Like I'm 5" Mode
// get_symbol("EmbeddingEngine", explain=true)
// Returns: "This is the thing that turns text into numbers so we can compare meanings"
LLM-generated plain-English summaries of what things do and WHY they exist.

8. Docpack Composition
doctown install axios  # grabs docpack from registry
doctown link ./my-lib.docpack  # local dev
Now my agent understands my dependencies' architecture too. NPM for knowledge.

The "Holy Shit" Feature
Conversational Query Interface:
doctown ask "How do I add a new MCP tool?"
# Returns: relevant symbols, files, examples, and a plain-English guide
Use the embeddings you already have + LLM to answer natural language questions about the codebase. This turns it from a lookup tool to an actual assistant.