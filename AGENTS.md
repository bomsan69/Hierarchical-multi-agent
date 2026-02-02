
This file provides guidance to Openc Code (claude.ai/code) when working with code in this repository.

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

## Code Style Guidelines

### Imports
- Place `from __future__ import annotations` at the top for forward references
- Organize imports: stdlib first, then third-party packages, separated by blank lines
- Use absolute imports from project modules

```python
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Annotated, List, Optional, TypedDict

from langchain_core.tools import tool
```

### Types and Type Hints
- Always use type hints for function parameters and return values
- Use `typing.Annotated` for tool parameters with descriptions
- Use `TypedDict` for structured data schemas
- Use `Optional[T]` for nullable types

```python
def extract_urls_from_text(text: str) -> list[str]:
    pass

class Router(TypedDict):
    next: str
```

### Naming Conventions
- **Classes**: PascalCase (`ReRankService`, `Config`, `ResearchGraphBuilder`)
- **Functions/Variables**: snake_case (`evaluate_relevance`, `agent_logger`)
- **Constants**: UPPER_CASE (`MAX_RETRIES`, `TIMEOUT`)
- **Private variables**: Prefix with underscore (`_repl`)
- **Type aliases**: PascalCase when exposing as public API

### Formatting
- Use 4 spaces for indentation (no tabs)
- Use double quotes for strings
- Use f-strings for string interpolation
- Line length: keep under 100 characters when practical
- Use blank lines to separate logical sections

### Error Handling
- Always log errors using `agent_logger.error()`
- Use try/except with specific exception types
- Return error messages as strings for tool functions
- Log both success and failure states for debugging

```python
try:
    result = _repl.run(code)
    agent_logger.info("[TOOL:python_repl] 코드 실행 성공")
    return f"Successfully executed:\n{result}"
except BaseException as e:
    agent_logger.error(f"[TOOL:python_repl] 코드 실행 실패: {repr(e)}")
    return f"Failed to execute. Error: {repr(e)}"
```

### Docstrings
- Use triple double quotes for docstrings
- Include `Args:` and `Returns:` sections for functions
- Write docstrings in Korean or English (maintain consistency within file)
- Keep docstrings concise but informative

```python
def evaluate_relevance(self, query: str, search_result: str) -> float:
    """
    검색 결과가 질문에 대해 얼마나 관련성이 있는지 평가합니다.

    Args:
        query: 사용자 질문
        search_result: 검색 결과

    Returns:
        관련성 점수 (0.0 ~ 1.0)
    """
```

### Dataclasses
- Use `@dataclass` for configuration and state objects
- Use `field(default_factory=...)` for mutable default values
- Document class purpose with docstring

```python
@dataclass
class Config:
    """애플리케이션 설정을 관리하는 클래스"""
    model_name: str = "gpt-4o"
    relevance_threshold: float = 0.5
    working_directory: Path = field(
        default_factory=lambda: Path(__file__).parent.resolve()
    )
```

### File Paths
- Use `pathlib.Path` for all file path operations
- Use `/` operator for path joining
- Use `.open()` context manager for file I/O with `encoding="utf-8"`

```python
file_path = config.working_directory / file_name
with file_path.open("w", encoding="utf-8") as file:
    file.write(content)
```

### Logging
- Use the global `agent_logger` instance for all logging
- Log at appropriate levels: DEBUG, INFO, WARNING, ERROR
- Include context in log messages (e.g., file names, operation results)
- Use structured prefixes for different components: `[COMPONENT]`

```python
agent_logger.info(f"[TOOL:create_outline] 아웃라인 생성 (포인트 수: {len(points)})")
agent_logger.error(f"[TOOL:create_outline] URL은 파일명으로 사용 불가: {file_name}")
```

### Section Dividers
- Use consistent section dividers for major code sections
- Format: `# ============ <SECTION_NAME> ============`

```python
# =============================================================================
# LOGGING CONFIGURATION
# =============================================================================
```

### Tools (LangChain)
- Use `@tool` decorator for LangChain tool functions
- Use `Annotated[Type, "Description"]` for tool parameters
- Return descriptive success/error messages as strings

```python
@tool
def scrape_webpages(urls: List[str]) -> str:
    """웹 페이지들을 스크래핑하여 상세 정보를 추출합니다."""
    pass
```

## Testing

Currently no test framework is configured. If adding tests:
- Use pytest as the test framework
- Place tests in `tests/` directory
- Run specific test: `uv run pytest tests/test_module.py::test_function`
- Run all tests: `uv run pytest`