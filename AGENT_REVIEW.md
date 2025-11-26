8. Docpack Composition
doctown install axios  # grabs docpack from registry
doctown link ./my-lib.docpack  # local dev
Now my agent understands my dependencies' architecture too. NPM for knowledge.

The "Holy Shit" Feature
Conversational Query Interface:
doctown ask "How do I add a new MCP tool?"
# Returns: relevant symbols, files, examples, and a plain-English guide
Use the embeddings you already have + LLM to answer natural language questions about the codebase. This turns it from a lookup tool to an actual assistant.