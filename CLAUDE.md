# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

A hierarchical multi-agent system built with LangGraph that coordinates research and document writing tasks. The system uses a three-tier architecture with a Super Graph managing specialized team subgraphs.

## Commands

```bash
# Install dependencies (uses uv package manager)
uv sync

# Run the application
uv run python main.py
```

## Architecture

### Three-Layer Hierarchy

1. **Super Graph** (top-level orchestrator)
   - Routes tasks between `research_team` and `writing_team`
   - Uses LLM-based supervisor for dynamic routing decisions

2. **Research Graph** (`research_team`)
   - `search`: Web search via Brave Search API with reRank quality evaluation
   - `web_scraper`: Scrapes URLs provided by user (not from search results)
   - Implements reRank pattern: evaluates search relevance (0.0-1.0), rephrases query if below 50% threshold

3. **Paper Writing Graph** (`writing_team`)
   - `doc_writer`: Creates and edits documents
   - `note_taker`: Creates outlines
   - `chart_generator`: Generates charts via Python REPL

### Key Classes

- `HierarchicalAgentTeams`: Main entry point in `main.py:851`
- `SuperGraphBuilder`: Builds top-level graph (`main.py:743`)
- `ResearchGraphBuilder`: Builds research subgraph with reRank (`main.py:464`)
- `PaperWritingGraphBuilder`: Builds writing subgraph (`main.py:672`)
- `ReRankService`: Evaluates search result quality (`main.py:254`)
- `Config`: Dataclass for application settings (`main.py:39`)

### State Types

- `State`: Base graph state with `messages` and `next` fields
- `ResearchState`: Extended state with `urls_found`, `search_completed`, `scrape_completed`, `relevance_score`, `search_result`

## Environment Variables

Requires `.env` file with:
- `OPENAI_API_KEY`: For ChatOpenAI (gpt-4o)
- `BRAVE_API_KEY`: For BraveSearch tool

## Configuration

Key settings in `Config` class:
- `relevance_threshold`: 0.5 (reRank minimum score)
- `max_retry_count`: 1 (search retries)
- `max_scrape_urls`: 3
- `recursion_limit`: 150

## Logging

로그 파일은 `logs/` 디렉토리에 `YYYY-MM-DD.agent.log` 형식으로 저장됩니다.

### 로그 레벨
- `DEBUG`: 상세 디버그 정보 (메시지 내용, 검색 결과 미리보기 등)
- `INFO`: 주요 이벤트 (노드 시작/완료, 라우팅 결정 등)
- `WARNING`: 재검색 시도, URL 없음 등 주의 사항
- `ERROR`: 스크래핑 실패, 파일 없음 등 오류

### 로그 포맷
```
YYYY-MM-DD HH:MM:SS | LEVEL    | agent | message
```

### 로그 대상
- Super Graph: `[RESEARCH_TEAM]`, `[WRITING_TEAM]`
- Research Graph: `[RESEARCH_SUPERVISOR]`, `[SEARCH_NODE]`, `[WEB_SCRAPER_NODE]`
- Writing Graph: `[WRITING/DOC_WRITER]`, `[WRITING/NOTE_TAKER]`, `[WRITING/CHART_GENERATOR]`
- 도구: `[TOOL:scrape_webpages]`, `[TOOL:write_document]` 등
- ReRank: `[ReRank]`
