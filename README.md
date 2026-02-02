Hierarchical Multi-Agent System
==============================

Hierarchical multi-agent orchestration playground built with LangGraph. The project wires
LLM-powered supervision, research tooling, and document production into a three-tier graph,
showcasing how separate specialist teams can coordinate through a top-level conductor.

Table of Contents
-----------------
- [Features](#features)
- [Project Structure](#project-structure)
- [Quick Start](#quick-start)
- [Running the Application](#running-the-application)
- [Configuration](#configuration)
- [Logging](#logging)
- [Development Guidelines](#development-guidelines)
- [Testing](#testing)
- [Troubleshooting](#troubleshooting)
- [Future Work](#future-work)

Features
--------
- Three-layer LangGraph architecture with a supervising graph that routes work between
  specialized research and writing teams.
- Research graph supports Brave Search queries, adaptive re-ranking, and optional web scraping
  for user-supplied URLs.
- Writing graph provides document authoring, outlining, and Python-based chart generation tools.
- Structured logging for every node/tool with agent-specific prefixes to simplify debugging.
- Configuration-driven behavior (model selection, retries, thresholds, limits) via a single
  dataclass and `.env` secrets.

Project Structure
-----------------
```
.
├── main.py              # LangGraph builders, tools, supervisors, entry point
├── pyproject.toml       # Project metadata & dependencies for UV
├── uv.lock              # Locked dependency versions
├── logs/                # Daily agent logs (ignored by git)
├── AGENTS.md            # Guidance for autonomous agents working in this repo
├── CLAUDE.md            # Legacy guidance for Claude-specific workflows
└── README.md            # You are here
```

Quick Start
-----------
1. Install UV (if not already installed) following <https://docs.astral.sh/uv/>
2. Clone the repository and enter the directory
3. Create a `.env` file with the required API keys (see [Configuration](#configuration))
4. Install dependencies:
   ```bash
   uv sync
   ```
5. Run the console demo:
   ```bash
   uv run python main.py
   ```

Running the Application
-----------------------
- `uv run python main.py` launches the hierarchical agent workflow. The supervisor graph will
  dynamically delegate tasks to the research or writing subgraphs based on the conversation state.
- Set the environment variable `LOG_LEVEL=DEBUG` before running if you want verbose console logs.
- Logs are always written to `logs/YYYY-MM-DD.agent.log`; ensure the directory is writable.

Configuration
-------------
All central settings live in the `Config` dataclass defined near the top of `main.py`. Key fields:
- `model_name`: OpenAI Chat model (default `gpt-4o`).
- `relevance_threshold`: Minimum acceptable relevancy score (0.5) for search results.
- `max_retry_count`: How many times reRank can rephrase and retry.
- `max_scrape_urls`: Safety limit for scraping external URLs (default 3).
- `request_timeout`: Timeout per scraping request in seconds (default 10).
- `recursion_limit`: Maximum graph recursion depth (default 150).

Environment Variables
~~~~~~~~~~~~~~~~~~~~~
Create a `.env` file with:
```
OPENAI_API_KEY=sk-...
BRAVE_API_KEY=brv-...
```
Both keys are required at runtime. Never commit `.env`.

Logging
-------
- Logs use the format `YYYY-MM-DD HH:MM:SS | LEVEL | agent | message`.
- Each component prefixes messages, e.g. `[RESEARCH_SUPERVISOR]`, `[TOOL:python_repl]`.
- Log files rotate daily and live under `logs/`. The directory is gitignored; create it manually if
  missing.

Development Guidelines
----------------------
- Always add `from __future__ import annotations` at the top of new Python modules.
- Group imports by standard library, third-party, and local modules with blank lines in between.
- Use `pathlib.Path` for all filesystem interactions and prefer `.open(..., encoding="utf-8")`.
- Favor dataclasses for structured configuration/state and annotate every field.
- Keep line length under ~100 characters and prefer f-strings for interpolation.
- Log errors through the shared `agent_logger`; never swallow exceptions silently.
- Use `typing.Annotated` for LangChain tool parameters so descriptions surface in tool schemas.
- Divide major sections with `# =============================================================================` banners for readability.

Testing
-------
The project currently ships without an automated test suite. If you add tests, follow this pattern:
- Install pytest (already available via `pyproject.toml` once added).
- Place tests under `tests/` with filenames like `test_research_graph.py`.
- Run the full suite:
  ```bash
  uv run pytest
  ```
- Run a single test case:
  ```bash
  uv run pytest tests/test_research_graph.py::test_rerank_threshold
  ```

Troubleshooting
---------------
- **Missing API keys**: Ensure `.env` contains valid `OPENAI_API_KEY` and `BRAVE_API_KEY` entries.
- **LangGraph import errors**: Run `uv sync` again; the lockfile pins working dependency versions.
- **Timeouts during scraping**: Adjust `Config.request_timeout` or reduce the number of URLs.
- **High recursion errors**: Increase `Config.recursion_limit` if you are chaining many operations.

Future Work
-----------
- Add pytest coverage for the routing logic and ReRank service.
- Provide a CLI or REST surface for triggering research/writing tasks without editing code.
- Incorporate structured output storage (e.g., Markdown exports or vector stores) for research notes.

License
-------
MIT-style licensing placeholder. Update this section with formal license text if distributing the
project publicly.
