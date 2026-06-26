# Housing QA Memory

## Retrieval
- Use BM25 before grepping the corpus.
- Start with small limits, usually 5 or 10.
- If a result looks relevant, inspect it with get-doc by doc_id.
- Use state or jurisdiction filters when the question identifies a state.
- Do not repeat the exact same search command; change the query, filter, or inspect a specific result.
- Do not increase the search limit just because output is truncated. If visible snippets or get-doc evidence are enough, answer.
- Deep tasks may need many searches, but each search should test a specific hypothesis.
