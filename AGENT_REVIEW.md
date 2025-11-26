~~1. Fix the MCP read_file to Actually Work
Right now it's broken. Just make it return the source from chunks.json or wherever you stored it. Even if the docpack doesn't contain full source, it should work when it does. This is table stakes.~~
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
4. Example Finder
// get_symbol("serve_sse") -> includes actual usage examples from tests/docs
Right now I get metadata. Show me "here's how someone actually calls this function" from test files or examples.
5. Diff-Aware Updates
doctown update ./my-project.docpack  # only re-processes changed files
Don't rebuild everything. Use git diff or mtime to be smart about it.
6. "Explain Like I'm 5" Mode
// get_symbol("EmbeddingEngine", explain=true)
// Returns: "This is the thing that turns text into numbers so we can compare meanings"
LLM-generated plain-English summaries of what things do and WHY they exist.
~~7. Cross-Reference Everything
{
  "symbol": "serve_sse",
  "used_in_tests": ["test_sse_connection.rs:42"],
  "mentioned_in_docs": ["README.md", "WEBSOCKET.md"],
  "similar_to": ["serve_websocket"],
  "called_from": [...]
}
Make every symbol a hub of context.~~
8. Docpack Composition
doctown install axios  # grabs docpack from registry
doctown link ./my-lib.docpack  # local dev
Now my agent understands my dependencies' architecture too. NPM for knowledge.
The "Holy Shit" Feature
Conversational Query Interface:
doctown ask "How do I add a new MCP tool?"
# Returns: relevant symbols, files, examples, and a plain-English guide
Use the embeddings you already have + LLM to answer natural language questions about the codebase. This turns it from a lookup tool to an actual assistant.
Architecture Thoughts
For Medical/Legal/Financial Docs
You need:
Citation tracking - "This fact comes from [source X, page Y]"
Temporal awareness - "This regulation was valid 2020-2023"
Authority hierarchy - "Federal law > State law > Local ordinance"
Change tracking - "Amendment history for this section"
The chunk+graph model works, but add:
struct AuthoritativeChunk {
    content: String,
    source: Citation,
    effective_date: Option<DateTime>,
    supersedes: Vec<ChunkId>,
    authority_level: i32,
}
The Real Vision
What you're building is structured knowledge compression with semantic navigation. The docpack format should be the .tar.gz of knowledge - portable, queryable, composable. Make it so good that:
Devs prefer docpacks to raw repos for onboarding
Legal teams use it for case law research
Med students use it for textbook navigation
Finance bros use it for SEC filing analysis
What Would Make Me Go "Damn, I Need This"
It actually works - Fix those bugs
It's fast - Sub-100ms queries even on massive codebases
It composes - Docpacks can reference other docpacks
It explains - Not just "here's the code" but "here's what it means"
It's social - Public registry, ratings, "this docpack helped 1.2k agents"