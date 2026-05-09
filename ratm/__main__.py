"""CLI entrypoint: ratm index / ratm query / ratm status."""
from __future__ import annotations

import json
import os
from pathlib import Path

import click
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

console = Console()


@click.group()
def cli() -> None:
    """RAG-against-the-machine: retrieval-augmented code Q&A."""


@cli.command()
@click.argument("path", type=click.Path(exists=True))
@click.option("--out", default="ratm-out", help="Output directory for index artifacts.")
@click.option("--summarize", is_flag=True, default=False, help="Generate Claude summaries at index time.")
def index(path: str, out: str, summarize: bool) -> None:
    """Index a repository or directory for retrieval."""
    import anthropic
    from ratm.index import index as run_index

    root = Path(path).resolve()
    out_dir = Path(out).resolve()

    client = None
    if summarize:
        api_key = os.environ.get("ANTHROPIC_API_KEY", "")
        if not api_key:
            console.print("[red]ANTHROPIC_API_KEY not set.[/red] Cannot generate summaries.")
            raise SystemExit(1)
        client = anthropic.Anthropic(api_key=api_key)

    console.print(f"[bold]Indexing[/bold] {root}")

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
        transient=True,
    ) as progress:
        task = progress.add_task("Extracting AST + embedding chunks...", total=None)
        stats = run_index(root, out_dir=out_dir, skip_summaries=not summarize, client=client)
        progress.update(task, completed=True)

    console.print(f"[green]Done.[/green] {stats}")
    console.print(f"Artifacts written to: {out_dir}")


@cli.command()
@click.argument("question")
@click.option("--out", default="ratm-out", help="Output directory where index lives.")
@click.option("--top-k", default=10, help="Number of chunks to pass to the LLM.")
def query(question: str, out: str, top_k: int) -> None:
    """Ask a question about the indexed codebase."""
    import anthropic
    from ratm.vectorstore import VectorStore
    from ratm.sparse import BM25Index
    from ratm.retrieve import retrieve
    from ratm.embed import embed_query
    from ratm.generate import generate_answer

    out_dir = Path(out).resolve()
    chroma_dir = out_dir / "chroma"
    bm25_dir = out_dir / "bm25"

    if not chroma_dir.exists():
        console.print("[red]No index found.[/red] Run `ratm index <path>` first.")
        raise SystemExit(1)

    vectorstore = VectorStore(persist_dir=str(chroma_dir))

    bm25 = BM25Index()
    if bm25_dir.exists():
        bm25.load(str(bm25_dir))

    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        console.print("[red]ANTHROPIC_API_KEY not set.[/red]")
        raise SystemExit(1)

    client = anthropic.Anthropic(api_key=api_key)

    graph_summary = _load_graph_summary(out_dir)
    community_summaries = _load_community_summaries(out_dir)

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
        transient=True,
    ) as progress:
        progress.add_task("Retrieving relevant chunks...", total=None)
        chunks = retrieve(
            query=question,
            vectorstore=vectorstore,
            bm25=bm25,
            embed_fn=embed_query,
            n_results=top_k,
            graph_path=str(out_dir / "graph.json") if (out_dir / "graph.json").exists() else None,
            hyde_client=client,
        )

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
        transient=True,
    ) as progress:
        progress.add_task("Generating answer...", total=None)
        answer = generate_answer(
            query=question,
            chunks=chunks,
            graph_summary=graph_summary,
            community_summaries=community_summaries,
            client=client,
        )

    console.print()
    console.rule("[bold]Answer[/bold]")
    console.print(answer)
    console.rule()

    if chunks:
        console.print(f"\n[dim]Sources ({len(chunks)} chunks):[/dim]")
        for chunk in chunks[:5]:
            meta = chunk.get("metadata", {})
            console.print(f"  [dim]{meta.get('source_file', '?')} {meta.get('source_location', '')}[/dim]")


@cli.command()
@click.option("--out", default="ratm-out", help="Output directory where index lives.")
def status(out: str) -> None:
    """Show index statistics."""
    out_dir = Path(out).resolve()
    chroma_dir = out_dir / "chroma"
    bm25_dir = out_dir / "bm25"
    graph_path = out_dir / "graph.json"

    console.print(f"[bold]ratm-out:[/bold] {out_dir}")

    if not out_dir.exists():
        console.print("[yellow]No index found.[/yellow]")
        return

    # Vector store count
    if chroma_dir.exists():
        from ratm.vectorstore import VectorStore
        vs = VectorStore(persist_dir=str(chroma_dir))
        console.print(f"  Vector store chunks: [cyan]{vs.count()}[/cyan]")
    else:
        console.print("  Vector store: [yellow]not found[/yellow]")

    # BM25
    if bm25_dir.exists():
        console.print(f"  BM25 index: [cyan]present[/cyan] ({bm25_dir})")
    else:
        console.print("  BM25 index: [yellow]not found[/yellow]")

    # Graph stats
    if graph_path.exists():
        try:
            data = json.loads(graph_path.read_text(encoding="utf-8"))
            n_nodes = len(data.get("nodes", []))
            n_edges = len(data.get("edges", data.get("links", [])))
            console.print(f"  Graph: [cyan]{n_nodes} nodes, {n_edges} edges[/cyan]")
        except Exception:
            console.print("  Graph: [yellow]present but unreadable[/yellow]")
    else:
        console.print("  Graph: [yellow]not found[/yellow]")


def _load_community_summaries(out_dir: Path) -> str:
    """Load community summaries from community_summaries.json if present."""
    path = out_dir / "community_summaries.json"
    if not path.exists():
        return "(community summaries not yet generated — run ratm index --summarize)"
    try:
        summaries: list[str] = json.loads(path.read_text(encoding="utf-8"))
        return "\n\n".join(s for s in summaries if s)
    except Exception:
        return "(community summaries unreadable)"


def _load_graph_summary(out_dir: Path) -> str:
    """Load graph summary from GRAPH_REPORT.md if present, else return placeholder."""
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


@cli.command()
@click.option("--out", default="ratm-out", help="Output directory where index lives.")
def serve(out: str) -> None:
    """Start the MCP server."""
    from ratm.serve import run_server
    run_server(out_dir=out)


@cli.command(name="install")
def install_cmd() -> None:
    """Install ratm skill and CLAUDE.md integration for Claude Code."""
    import importlib.resources
    import shutil

    skill_dst = Path.home() / ".claude" / "skills" / "ratm"
    skill_dst.mkdir(parents=True, exist_ok=True)
    dst_file = skill_dst / "SKILL.md"

    # Copy bundled skill.md
    try:
        ref = importlib.resources.files("ratm").joinpath("skill.md")
        with importlib.resources.as_file(ref) as src:
            shutil.copy2(src, dst_file)
        console.print(f"[green]✓[/green] Skill installed: {dst_file}")
    except Exception as e:
        console.print(f"[red]✗[/red] Skill install failed: {e}")
        raise SystemExit(1)

    # Append blurb to ~/.claude/CLAUDE.md if not already present
    claude_md = Path.home() / ".claude" / "CLAUDE.md"
    marker = "<!-- ratm -->"
    if claude_md.exists() and marker in claude_md.read_text(encoding="utf-8"):
        console.print("[dim]CLAUDE.md already contains ratm section — skipped[/dim]")
    else:
        blurb = f"""
{marker}
## ratm — Code Q&A

ratm indexes codebases for retrieval-augmented Q&A. Use it instead of reading files when:
- Question spans multiple files or modules
- Codebase is too large for context
- Same repo queried repeatedly (prompt caching = cheap)

```bash
ratm index <path>          # build index
ratm query "<question>"    # ask a question with cited answer
ratm status                # check index
ratm serve                 # start MCP server
```

Full docs: see ratm/skill.md or run `ratm --help`.
<!-- /ratm -->
"""
        with claude_md.open("a", encoding="utf-8") as f:
            f.write(blurb)
        console.print(f"[green]✓[/green] CLAUDE.md updated: {claude_md}")


if __name__ == "__main__":
    cli()
