# Checkpoint — RAG-against-the-machine

## What this project is

Replacing graphify with a real retrieval + generation system. Graphify builds an AST knowledge graph but has no query interface — you can browse the graph but can't ask it questions. This adds vector indexing, hybrid search, graph-guided retrieval (PPR), and LLM answer generation on top of graphify's extraction pipeline.

**CLI name:** `ratm`  
**Repo:** `/Users/jammin/Documents/GitHub/RAG-against-the-machine/`  
**Upstream dependency:** `/Users/jammin/Documents/GitHub/graphify/` (package name is `graphifyy` — two y's)

---

## Architecture (full target)

```
Index time (once, incremental):
  tree-sitter AST parse      → nodes + edges          (graphify.extract)
  BM25 index                 → sparse term index       (bm25s)
  BGE-M3 embed chunks        → vector store            (ChromaDB)
  Claude: summarize chunks   → chunk summaries         (disk cache, Phase 3)
  Leiden community detection → community labels        (graphify.cluster)
  Claude: summarize communities → community summaries  (disk cache, Phase 3)
  GRAPH_REPORT.md            → global structural map   (graphify.report)

Query time (per question):
  1. HyDE: Claude generates hypothetical answer doc    (Phase 3)
  2. Embed hypothetical + original query (BGE-M3 asymmetric)
  3. Dense search (ChromaDB) → top-50 candidates
  4. BM25 search → top-50 candidates
  5. PPR on graph from seed nodes → related nodes      (Phase 2)
  6. Merge all candidates (RRF)
  7. Cross-encoder rerank (BGE-reranker, local)        (Phase 2)
  8. Assemble tiered context (cached prefix + dynamic chunks)
  9. Claude API (with prompt caching) → answer + citations
```

### Two-model problem mitigations (inference-only, no training)
- **HyDE**: Claude generates hypothetical doc → embed in doc-space (bridges retriever ≠ generator space)
- **Cross-encoder reranking**: BGE-reranker-v2-m3 scores (query, doc) pairs jointly
- **Graph PPR**: Objective AST structure acts as model-agnostic bridge
- **Prompt caching**: Tiered context — cached static prefix (system + graph summary + community summaries) + dynamic chunks per query
- **Asymmetric embeddings**: BGE-M3 instruction prefixes for query vs doc

### CLaRa paper concepts → local equivalents
| CLaRa (trains models) | Our equivalent (inference-only) |
|---|---|
| SCP: compress docs into memory tokens | Precompute chunk summaries at index time (Claude, disk-cached) |
| Query reasoner (NTP-trained) | HyDE: Claude generates hypothetical answer doc |
| Differentiable top-k | Cross-encoder reranking (BGE local) |
| Shared latent space | Graph PPR bridges retriever + generator |
| Joint optimization | Prompt caching aligns context to generator needs |

---

## Phase status

### ✅ Phase 1 — Core pipeline (COMPLETE)
All files installed and verified with a synthetic test corpus (5 nodes, retrieval working).

| File | Status |
|---|---|
| `pyproject.toml` | ✅ installed (`pip install -e .`) |
| `ratm/__init__.py` | ✅ |
| `ratm/__main__.py` | ✅ CLI: `ratm index`, `ratm query`, `ratm status` |
| `ratm/cache.py` | ✅ content-hash disk cache |
| `ratm/embed.py` | ✅ BGE-M3 lazy-loaded, asymmetric prefixes |
| `ratm/vectorstore.py` | ✅ ChromaDB persistent |
| `ratm/sparse.py` | ✅ BM25 with directory save + ids.json sidecar |
| `ratm/retrieve.py` | ✅ dense + BM25, RRF fusion |
| `ratm/generate.py` | ✅ Claude with 3-tier prompt caching |
| `ratm/index.py` | ✅ full AST → embed → ChromaDB + BM25 → graph JSON |

### 🔲 Phase 2 — Graph-guided retrieval
- `ratm/graph.py` — PPR on scipy sparse adjacency; `ppr(G_sparse, seed_ids, alpha=0.85) → {node_id: score}`; build sparse matrix from `ratm-out/graph.json` at query time
- `ratm/rerank.py` — BGE cross-encoder (`BAAI/bge-reranker-v2-m3`); `rerank(query, candidates) → top_k`
- Update `ratm/retrieve.py` — add PPR + reranking step after RRF merge

### 🔲 Phase 3 — Quality improvements (SCP equivalent)
- `ratm/summarize.py` — `summarize_chunk(node, client) → str` (Claude, disk-cached by content hash); `summarize_community(nodes, edges, client) → str`
- Update `ratm/index.py` — call summarize_chunk for each node at index time (skip if cached)
- Update `ratm/generate.py` — add `hyde_expand(query, client) → str`; embed hypothetical instead of raw query; pass real community summaries

### 🔲 Phase 4 — MCP server
- `ratm/serve.py` — extend graphify's serve.py MCP pattern; expose `query`, `index_status`, `graph_stats` tools

---

## Key discoveries / corrections

1. **graphify package name is `graphifyy`** (two y's) — actual name in graphify's pyproject.toml
2. **`collect_files` + `extract` both live in `graphify.extract`**, not `graphify.detect`
3. **`extract()` takes `list[Path]`**, not a single path — call once over all files
4. **bm25s saves a directory**, not a `.pkl` file — needs `ids.json` sidecar for id mapping
5. **ChromaDB batch limit** — add chunks in batches of 500 to avoid memory issues
6. **`graphify.extract.extract()` signature**: `extract(files: list[Path], cache_root: Path) -> dict` (returns single merged extraction dict)

---

## Output structure

```
ratm-out/
├── cache/
│   ├── ast/{hash}.json         # AST extraction cache
│   └── summaries/{hash}.json   # chunk + community summaries (Phase 3)
├── chroma/                     # ChromaDB vector store
├── bm25/                       # BM25 index directory
│   └── ids.json                # id mapping sidecar
├── graph.json                  # full graph (networkx node_link_data format)
└── GRAPH_REPORT.md             # global structural map (Phase 3+)
```

---

## To resume work

```bash
cd /Users/jammin/Documents/GitHub/RAG-against-the-machine

# Verify install
ratm --help

# Test index on graphify itself
ratm index /Users/jammin/Documents/GitHub/graphify

# Test query
ratm query "how does file caching work?"
ratm query "what functions handle AST extraction?"

# Then build Phase 2:
# 1. ratm/graph.py (PPR)
# 2. ratm/rerank.py (BGE cross-encoder)
# 3. update ratm/retrieve.py to wire both in
```

---

## Design decisions (why)

- **ChromaDB over FAISS**: no separate process, persists to disk, simpler API
- **BGE-M3 over nomic-embed-text**: handles code + text, asymmetric instruction support, single model for both
- **Claude API for generation**: best quality; prompt caching makes repeated queries cheap
- **RRF over learned fusion**: no training needed, proven in literature, robust to score scale differences
- **PPR alpha=0.85**: standard damping factor; biases toward graph structure over pure random walk
- **Graphify as dependency**: reuse AST extraction, caching, clustering, analysis — don't rewrite
- **Tiered prompt caching**: system + graph summary + community summaries cached; chunks + query dynamic — ~90% token savings on repeated queries to same module
