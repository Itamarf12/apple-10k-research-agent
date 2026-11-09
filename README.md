# Apple 10-K Research Agent

A command-line agent that answers questions about Apple's latest 10-K (fiscal year ended
September 27, 2025). Gemini decides which tools to call: **Document QA** over the filing,
**Tavily** web search, and a **calculator**. Every claim is cited.

## Setup

1. Clone the repository and create a virtual environment (Python 3.10+).
2. Install dependencies: `pip install -r requirements.txt`
3. Download Apple's 10-K from SEC EDGAR and place it in `data/` as `apple_10k_2025.htm`
   (already included). Source: the "Documents" page of the 10-K filed 2025-10-31,
   file `aapl-20250927.htm`:
   https://www.sec.gov/Archives/edgar/data/320193/000032019325000079/0000320193-25-000079-index.htm
4. Copy `.env.example` to `.env` and add your keys:
   - `GEMINI_API_KEY` (free, no credit card: https://aistudio.google.com)
   - `TAVILY_API_KEY` (free tier, 1,000 requests a month: https://tavily.com)
5. Build the search index once: `python src/build_index.py`
   (the first run downloads Chroma's small embedding model, about 80 MB).
6. Run: `python src/main.py`

```
$ python src/main.py
> What are Apple's top 3 risk factors?
[Agent processing...]
[Plan]
1. document_qa: search Item 1A ...
[Tool: document_qa] {'query': 'Item 1A Macroeconomic and Industry Risks'}
...
```

One question and exit: `python src/main.py -q "What was Apple's R&D expense in fiscal 2025?"`

## How to run the test queries

The two required queries are in `tests/test_queries.py`. They call the real Gemini and Tavily
APIs, so they need the keys in `.env` and the built index:

```
pytest tests/test_queries.py -s
```

`-s` prints the tool trace and the answer for each query.

- **Query 1 (document only):** top 3 risk factors and R&D as a share of revenue.
  Expected tools: `document_qa` (Item 1A), `document_qa` (financial statements), `calculator`.
  Expected figure: 34,550 / 416,161 = 8.30%.
- **Query 2 (document + web):** Apple vs. Microsoft gross margin and Apple's reasons for margin
  pressure. Expected tools: `document_qa` (gross margin 46.9%, MD&A reasons), `tavily_search`
  (Microsoft), `calculator`.

Offline tests (no keys, no network except the first embedding download) check the parser,
chunker, cross-references, calculator, retrieval and the agent loop with a scripted model:

```
pytest tests/test_offline.py
```

## Architecture overview

```
question -> Agent --plan--> Gemini <--> tools  (function calling loop, max 8 steps)
                                         |- document_qa    -> Chroma index of the 10-K + tables.json
                                         |- tavily_search  -> Tavily API
                                         '- calculator     -> safe arithmetic
         -> answer with citations
```

- **How tools are registered.** `prompts/tool_descriptions.txt` holds each tool's name and
  description. `src/tools/registry.py` reads it and builds the Gemini function declarations and a
  name -> function dictionary.
- **How the agent decides which tool to use.** Gemini chooses from the tool descriptions and the
  rules in `prompts/system_prompt.txt` (10-K first, web only for live data, calculator for all
  math). A short planning step (`prompts/plan_prompt.txt`) prints the intended order first.
- **How information flows.** `src/agent.py` sends the question and plan to Gemini. When Gemini
  asks for tools, the code runs them (errors are returned as text, never raised) and sends the
  results back. This repeats until Gemini answers without requesting a tool.
- **Where prompts are stored.** All in `prompts/`: `system_prompt.txt`, `plan_prompt.txt`,
  `tool_descriptions.txt`, `examples.txt`. Nothing is hard-coded in Python.

### Document QA design

| Question | Approach |
|---|---|
| Tables vs. text | Tables are one chunk each, never split, shown to the LLM as Markdown with their header rows. Text is grouped by paragraphs up to about 170 words (the embedding model reads at most 256 tokens, so larger chunks would be cut off). EDGAR's split cells (`$`, `%`) are merged back. |
| Section context | Parser tracks Item, Note and heading. Every chunk is embedded with a location line such as `[Apple 10-K FY2025 › Item 7 – MD&A › Gross Margin]` and stores `item`, `note`, `heading_path`, `page`. |
| Chunking | Structure first (Item, Note, heading), then paragraphs. Never crosses a heading. |
| Related chunks | A table and the text under the same heading are linked both ways (`linked_ids`); retrieval returns them together. |
| Cross-references | "Note N", "Item N" and phrases such as "under the heading Risk Factors" are extracted into `refs`. At search time the referenced section is searched with the same query and the best match is added. |
| Entities | Apple's segments and product categories are matched into an `entities` field. |
| Narrative <-> table | Same-heading links, see above (for example the Gross Margin text and table). |
| Metadata | `type, seq, item, note, heading_path, page, entities, refs, linked_ids, citation`. |

Page numbers come from the page footers ("Apple Inc. | 2025 Form 10-K | 24"). Citations look like
`[10-K FY2025, Item 7 – MD&A, Gross Margin, p. 24]`. Web citations are `[Tavily: title, url, retrieved date]`.

Putting `Item 1A` or `Note 7` in a `document_qa` query restricts the search to that section.

### Error handling

Gemini rate limits and server errors are retried (2, 4, 8 seconds). Tavily errors, an unbuilt
index, missing keys, unknown tools, bad calculator input and the step limit are all reported as
plain text so the agent can retry or tell the user. Everything is logged to `logs/agent.log`.

## Project layout

```
src/main.py            CLI
src/agent.py           planning step + function-calling loop
src/build_index.py     builds Chroma + data/tables.json
src/config.py          paths and tuning constants
src/tools/             document_qa.py, tavily_search.py, calculator.py, registry.py
src/utils/             parser.py, indexer.py, store.py, embeddings.py, llm.py, logging_setup.py
prompts/               system_prompt.txt, plan_prompt.txt, tool_descriptions.txt, examples.txt
data/                  apple_10k_2025.htm (input); chroma/ and tables.json (generated)
tests/                 test_queries.py (2 required queries), test_offline.py
```

## Dependencies

| Package | Why |
|---|---|
| `google-genai` | Gemini API with function calling (free tier) |
| `tavily-python` | Required web search tool |
| `chromadb` | Local vector database, no server; includes a free local embedding model |
| `beautifulsoup4`, `lxml` | Parse the 10-K HTML and its tables |
| `python-dotenv` | Load keys from `.env` (never hard-coded) |
| `pytest` | Tests |

## Configuration

- `GEMINI_MODEL` (optional) overrides the default `gemini-3.8-flash`.
  The outputs in `examples/` were produced with `gemini-3.5-flash-lite`.
- `CHROMA_DIR` (optional) moves the vector store, for disks that do not support SQLite.
- `EMBEDDING_BACKEND=offline` uses a small hashing embedder instead of downloading the model. It
  exists for tests on machines without internet; search quality is lower.

## Known limitations

- Vector search only (no keyword index). Exact terms can be missed.
- The entity list and the reference phrases are written by hand for Apple's filing.
- Apple does not rank its risk factors, so "top 3" means the first three in Item 1A, and the agent says so.
- Web figures can cover a different period than the 10-K; the prompt asks the agent to state both.
