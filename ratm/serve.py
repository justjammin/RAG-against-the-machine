"""MCP stdio server — exposes RAG query tools to Claude and other agents."""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path


def _load_graph_summary(out_dir: Path) -> str:
    report = out_dir / "GRAPH_REPORT.md"
    if report.exists():
        try:
            return report.read_text(encoding="utf-8")[:4000]
        except Exception:
            pass
    graph_path = out_dir / "graph.json"
    if graph_path.exists():
        try:
            data = json.loads(graph_path.read_text(encoding="utf-8"))
            n_nodes = len(data.get("nodes", []))
            n_edges = len(data.get("edges", data.get("links", [])))
            return f"Graph: {n_nodes} nodes, {n_edges} edges."
        except Exception:
            pass
    return "(graph summary not available)"


def _load_community_summaries(out_dir: Path) -> str:
    summaries_path = out_dir / "community_summaries.json"
    if not summaries_path.exists():
        return "(community summaries not yet generated)"
    try:
        data = json.loads(summaries_path.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            parts = [f"Community {cid}:\n{summary}" for cid, summary in data.items()]
            return "\n\n".join(parts)
        if isinstance(data, list):
            return "\n\n".join(str(item) for item in data)
        return str(data)
    except Exception:
        return "(community summaries unreadable)"


def _filter_blank_stdin() -> None:
    import os
    import threading

    r_fd, w_fd = os.pipe()
    saved_fd = os.dup(sys.stdin.fileno())

    def _relay() -> None:
        try:
            with open(saved_fd, "rb") as src, open(w_fd, "wb") as dst:
                for line in src:
                    if line.strip():
                        dst.write(line)
                        dst.flush()
        except Exception:
            pass

    threading.Thread(target=_relay, daemon=True).start()
    os.dup2(r_fd, sys.stdin.fileno())
    os.close(r_fd)
    sys.stdin = open(0, "r", closefd=False)


def run_server(out_dir: str = "ratm-out") -> None:
    """Start the MCP server. Requires pip install mcp."""
    try:
        from mcp.server import Server
        from mcp.server.stdio import stdio_server
        from mcp import types
    except ImportError as exc:
        raise ImportError("mcp not installed. Run: pip install mcp") from exc

    server = Server("ratm")

    @server.list_tools()
    async def list_tools() -> list[types.Tool]:
        return [
            types.Tool(
                name="query",
                description="Ask a question about the indexed codebase. Runs hybrid RAG retrieval and returns a generated answer.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "question": {"type": "string", "description": "Natural language question about the codebase"},
                        "out_dir": {"type": "string", "default": "ratm-out", "description": "Directory where index artifacts live"},
                        "top_k": {"type": "integer", "default": 10, "description": "Number of chunks to retrieve and pass to the LLM"},
                    },
                    "required": ["question"],
                },
            ),
            types.Tool(
                name="index_status",
                description="Return index statistics: chunk count, BM25 presence, graph size, community summaries.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "out_dir": {"type": "string", "default": "ratm-out", "description": "Directory where index artifacts live"},
                    },
                },
            ),
            types.Tool(
                name="graph_stats",
                description="Return graph statistics: node count, edge count, community count.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "out_dir": {"type": "string", "default": "ratm-out", "description": "Directory where index artifacts live"},
                    },
                },
            ),
        ]

    def _tool_query(arguments: dict) -> str:
        import anthropic
        from ratm.vectorstore import VectorStore
        from ratm.sparse import BM25Index
        from ratm.retrieve import retrieve
        from ratm.embed import embed_query
        from ratm.generate import generate_answer

        question = arguments["question"]
        base = Path(arguments.get("out_dir", out_dir)).resolve()
        top_k = int(arguments.get("top_k", 10))

        chroma_dir = base / "chroma"
        if not chroma_dir.exists():
            return "No index found. Run `ratm index <path>` first."

        api_key = os.environ.get("ANTHROPIC_API_KEY", "")
        if not api_key:
            return "ANTHROPIC_API_KEY environment variable not set."

        vectorstore = VectorStore(persist_dir=str(chroma_dir))

        bm25 = BM25Index()
        bm25_dir = base / "bm25"
        if bm25_dir.exists():
            bm25.load(str(bm25_dir))

        graph_path_str: str | None = None
        graph_path = base / "graph.json"
        if graph_path.exists():
            graph_path_str = str(graph_path)

        client = anthropic.Anthropic(api_key=api_key)

        graph_summary = _load_graph_summary(base)
        community_summaries = _load_community_summaries(base)

        chunks = retrieve(
            query=question,
            vectorstore=vectorstore,
            bm25=bm25,
            embed_fn=embed_query,
            n_results=top_k,
            graph_path=graph_path_str,
            hyde_client=client,
        )

        return generate_answer(
            query=question,
            chunks=chunks,
            graph_summary=graph_summary,
            community_summaries=community_summaries,
            client=client,
        )

    def _tool_index_status(arguments: dict) -> str:
        from ratm.vectorstore import VectorStore

        base = Path(arguments.get("out_dir", out_dir)).resolve()
        chroma_dir = base / "chroma"
        bm25_dir = base / "bm25"
        graph_path = base / "graph.json"
        summaries_path = base / "community_summaries.json"

        chunk_count = 0
        if chroma_dir.exists():
            vs = VectorStore(persist_dir=str(chroma_dir))
            chunk_count = vs.count()

        has_bm25 = bm25_dir.exists()

        graph_nodes = 0
        graph_edges = 0
        if graph_path.exists():
            try:
                data = json.loads(graph_path.read_text(encoding="utf-8"))
                graph_nodes = len(data.get("nodes", []))
                graph_edges = len(data.get("edges", data.get("links", [])))
            except Exception:
                pass

        has_summaries = summaries_path.exists()

        result = {
            "chunks": chunk_count,
            "bm25": has_bm25,
            "graph_nodes": graph_nodes,
            "graph_edges": graph_edges,
            "has_summaries": has_summaries,
        }
        return json.dumps(result)

    def _tool_graph_stats(arguments: dict) -> str:
        base = Path(arguments.get("out_dir", out_dir)).resolve()
        graph_path = base / "graph.json"
        summaries_path = base / "community_summaries.json"

        if not graph_path.exists():
            return json.dumps({"nodes": 0, "edges": 0, "communities": 0})

        try:
            data = json.loads(graph_path.read_text(encoding="utf-8"))
        except Exception:
            return json.dumps({"error": "graph.json unreadable"})

        nodes = len(data.get("nodes", []))
        edges = len(data.get("edges", data.get("links", [])))

        communities = 0
        if summaries_path.exists():
            try:
                sdata = json.loads(summaries_path.read_text(encoding="utf-8"))
                if isinstance(sdata, dict):
                    communities = len(sdata)
                elif isinstance(sdata, list):
                    communities = len(sdata)
            except Exception:
                pass

        return json.dumps({"nodes": nodes, "edges": edges, "communities": communities})

    _handlers = {
        "query": _tool_query,
        "index_status": _tool_index_status,
        "graph_stats": _tool_graph_stats,
    }

    @server.call_tool()
    async def call_tool(name: str, arguments: dict) -> list[types.TextContent]:
        handler = _handlers.get(name)
        if not handler:
            return [types.TextContent(type="text", text=f"Unknown tool: {name}")]
        try:
            return [types.TextContent(type="text", text=handler(arguments))]
        except Exception as exc:
            return [types.TextContent(type="text", text=f"Error executing {name}: {exc}")]

    import asyncio

    async def main() -> None:
        async with stdio_server() as streams:
            await server.run(streams[0], streams[1], server.create_initialization_options())

    _filter_blank_stdin()
    asyncio.run(main())


if __name__ == "__main__":
    run_server()
