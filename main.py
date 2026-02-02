"""
Hierarchical Agent Teams with reRank

이 모듈은 LangGraph를 사용한 계층적 에이전트 팀 시스템을 구현합니다.
- Super Graph: 최상위 조율자 (research_team, writing_team 관리)
- Research Graph: 웹 검색 및 스크래핑 (reRank 기반 품질 평가)
- Paper Writing Graph: 문서 작성, 아웃라인, 차트 생성

Author: AI Assistant
Version: 1.0.0
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Annotated, Any, Callable, Dict, List, Literal, Optional, TypedDict

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.tools import BraveSearch
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from langchain_experimental.utilities import PythonREPL
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.types import Command

load_dotenv()


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class Config:
    """애플리케이션 설정을 관리하는 클래스"""

    # 모델 설정
    model_name: str = "gpt-4o"

    # reRank 설정
    relevance_threshold: float = 0.5  # 만족도 임계값 (50%)
    max_retry_count: int = 1  # 재검색 최대 횟수

    # 웹 스크래핑 설정
    max_scrape_urls: int = 3  # 최대 스크래핑 URL 수
    request_timeout: int = 10  # 요청 타임아웃 (초)

    # 파일 시스템 설정
    working_directory: Path = field(
        default_factory=lambda: Path(__file__).parent.resolve()
    )

    # 실행 설정
    recursion_limit: int = 150


# 전역 설정 인스턴스
config = Config()


# =============================================================================
# TYPE DEFINITIONS
# =============================================================================

class State(MessagesState):
    """기본 그래프 상태"""
    next: str


class ResearchState(MessagesState):
    """Research Graph 전용 상태 (reRank 기능 포함)"""
    next: str
    urls_found: list[str]       # 사용자가 제공한 URL 목록 (web_scraper에서 스크랩)
    search_completed: bool      # search 완료 여부
    scrape_completed: bool      # scrape 완료 여부
    relevance_score: float      # reRank 만족도 점수 (0.0 ~ 1.0)
    search_result: str          # 검색 결과 저장


class RelevanceScore(TypedDict):
    """검색 결과 관련성 평가 스키마"""
    score: float      # 0.0 ~ 1.0 사이의 만족도 점수
    reasoning: str    # 점수 부여 이유


class RephrasedQuery(TypedDict):
    """재구성된 쿼리 스키마"""
    new_query: str    # 재구성된 검색어
    reasoning: str    # 재구성 이유


class Router(TypedDict):
    """Supervisor 라우팅 스키마"""
    next: str         # 다음에 실행할 워커 이름


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def extract_urls_from_text(text: str) -> list[str]:
    """
    텍스트에서 URL을 추출합니다.

    Args:
        text: URL을 추출할 텍스트

    Returns:
        추출된 URL 목록
    """
    url_pattern = r'https?://[^\s<>"{}|\\^`\[\]]+'
    return re.findall(url_pattern, text)


# =============================================================================
# TOOLS - Document Operations
# =============================================================================

@tool
def create_outline(
    points: Annotated[List[str], "List of main points or sections."],
    file_name: Annotated[str, "File path to save the outline. Must be a local file name, not a URL."],
) -> Annotated[str, "Path of the saved outline file."]:
    """아웃라인을 생성하고 파일로 저장합니다. URL은 파일명으로 사용할 수 없습니다."""
    # URL인 경우 오류 반환
    if file_name.startswith(("http://", "https://")):
        return f"Error: '{file_name}' is a URL, not a valid file name."

    file_path = config.working_directory / file_name
    with file_path.open("w", encoding="utf-8") as file:
        for i, point in enumerate(points, 1):
            file.write(f"{i}. {point}\n")
    return f"Outline saved to {file_name}"


@tool
def read_document(
    file_name: Annotated[str, "File path to read the document from. Must be a local file name, not a URL."],
    start: Annotated[Optional[int], "The start line. Default is 0"] = None,
    end: Annotated[Optional[int], "The end line. Default is None"] = None,
) -> str:
    """지정된 문서를 읽어 내용을 반환합니다. URL은 지원하지 않습니다."""
    # URL인 경우 오류 반환
    if file_name.startswith(("http://", "https://")):
        return f"Error: '{file_name}' is a URL, not a local file. Use scrape_webpages tool for URLs."

    file_path = config.working_directory / file_name

    if not file_path.exists():
        return f"Error: File '{file_name}' does not exist."

    with file_path.open("r", encoding="utf-8") as file:
        lines = file.readlines()
    start = start or 0
    return "\n".join(lines[start:end])


@tool
def write_document(
    content: Annotated[str, "Text content to be written into the document."],
    file_name: Annotated[str, "File path to save the document. Must be a local file name, not a URL."],
) -> Annotated[str, "Path of the saved document file."]:
    """텍스트 문서를 생성하고 저장합니다. URL은 파일명으로 사용할 수 없습니다."""
    # URL인 경우 오류 반환
    if file_name.startswith(("http://", "https://")):
        return f"Error: '{file_name}' is a URL, not a valid file name."

    file_path = config.working_directory / file_name
    with file_path.open("w", encoding="utf-8") as file:
        file.write(content)
    return f"Document saved to {file_name}"


@tool
def edit_document(
    file_name: Annotated[str, "Path of the document to be edited. Must be a local file name, not a URL."],
    inserts: Annotated[
        Dict[int, str],
        "Dictionary where key is the line number (1-indexed) and value is the text to be inserted.",
    ],
) -> Annotated[str, "Path of the edited document file."]:
    """문서의 특정 라인에 텍스트를 삽입하여 편집합니다. URL은 지원하지 않습니다."""
    # URL인 경우 오류 반환
    if file_name.startswith(("http://", "https://")):
        return f"Error: '{file_name}' is a URL, not a local file."

    file_path = config.working_directory / file_name

    if not file_path.exists():
        return f"Error: File '{file_name}' does not exist."

    with file_path.open("r", encoding="utf-8") as file:
        lines = file.readlines()

    for line_number, text in sorted(inserts.items()):
        if 1 <= line_number <= len(lines) + 1:
            lines.insert(line_number - 1, text + "\n")
        else:
            return f"Error: Line number {line_number} is out of range."

    with file_path.open("w", encoding="utf-8") as file:
        file.writelines(lines)

    return f"Document edited and saved to {file_name}"


# =============================================================================
# TOOLS - Web Operations
# =============================================================================

@tool
def scrape_webpages(urls: List[str]) -> str:
    """
    웹 페이지들을 스크래핑하여 상세 정보를 추출합니다.

    Args:
        urls: 스크래핑할 URL 목록

    Returns:
        스크래핑된 문서 내용들
    """
    results = []

    for url in urls[:config.max_scrape_urls]:
        try:
            loader = WebBaseLoader(
                url,
                requests_kwargs={"timeout": config.request_timeout}
            )
            docs = loader.load()
            for doc in docs:
                title = doc.metadata.get("title", "")
                results.append(
                    f'<Document name="{title}">\n{doc.page_content}\n</Document>'
                )
        except Exception as e:
            results.append(
                f'<Document url="{url}">\nError scraping page: {str(e)}\n</Document>'
            )

    return "\n\n".join(results)


# =============================================================================
# TOOLS - Code Execution
# =============================================================================

_repl = PythonREPL()


@tool
def python_repl_tool(
    code: Annotated[str, "The python code to execute to generate your chart."],
) -> str:
    """
    Python 코드를 실행합니다.

    WARNING: 로컬에서 코드를 실행하므로 샌드박스 환경이 아닐 경우 위험할 수 있습니다.
    """
    try:
        result = _repl.run(code)
        return f"Successfully executed:\n```python\n{code}\n```\nStdout: {result}"
    except BaseException as e:
        return f"Failed to execute. Error: {repr(e)}"


# =============================================================================
# RERANK SERVICE
# =============================================================================

class ReRankService:
    """
    검색 결과의 품질을 평가하고 필요시 재검색을 수행하는 서비스

    reRank 개념:
    - 검색 결과가 질문에 얼마나 적합한지 0.0~1.0 점수로 평가
    - 만족도가 임계값(50%) 미만이면 질문을 재구성하여 재검색
    - 두 결과 중 더 높은 만족도의 결과를 선택
    """

    def __init__(self, llm: BaseChatModel, threshold: float = 0.5):
        """
        Args:
            llm: 평가에 사용할 언어 모델
            threshold: 만족도 임계값 (기본 0.5)
        """
        self.llm = llm
        self.threshold = threshold

    def evaluate_relevance(self, query: str, search_result: str) -> float:
        """
        검색 결과가 질문에 대해 얼마나 관련성이 있는지 평가합니다.

        Args:
            query: 사용자 질문
            search_result: 검색 결과

        Returns:
            관련성 점수 (0.0 ~ 1.0)
        """
        evaluation_prompt = f"""You are a relevance evaluator. Evaluate how well the search results answer the user's question.

User's Question: {query}

Search Results:
{search_result}

Rate the relevance from 0.0 to 1.0:
- 0.0-0.3: Search results are irrelevant or don't address the question
- 0.3-0.5: Search results are partially relevant but missing key information
- 0.5-0.7: Search results adequately answer the question with some gaps
- 0.7-0.9: Search results provide good, comprehensive information
- 0.9-1.0: Search results perfectly answer the question with detailed information

Return your score and reasoning."""

        response = self.llm.with_structured_output(RelevanceScore).invoke([
            {"role": "user", "content": evaluation_prompt}
        ])

        return response["score"]

    def rephrase_query(self, original_query: str, search_result: str) -> str:
        """
        검색 결과가 부족할 때 질문을 재구성합니다.

        Args:
            original_query: 원래 질문
            search_result: 불충분한 검색 결과

        Returns:
            재구성된 질문
        """
        rephrase_prompt = f"""The original search query did not yield satisfactory results. Please rephrase the query to get better search results.

Original Query: {original_query}

Previous Search Results (unsatisfactory):
{search_result[:1000]}...

Create a new, improved search query that:
1. Uses different keywords or phrasing
2. Is more specific or broader as needed
3. Focuses on the key information the user needs

Return the rephrased query."""

        response = self.llm.with_structured_output(RephrasedQuery).invoke([
            {"role": "user", "content": rephrase_prompt}
        ])

        return response["new_query"]

    def should_retry(self, score: float) -> bool:
        """만족도 점수가 임계값 미만인지 확인합니다."""
        return score < self.threshold

    def is_satisfactory(self, score: float) -> bool:
        """만족도 점수가 임계값 이상인지 확인합니다."""
        return score >= self.threshold


# =============================================================================
# AGENT FACTORY
# =============================================================================

class AgentFactory:
    """에이전트 생성을 담당하는 팩토리 클래스"""

    def __init__(self, llm: BaseChatModel):
        self.llm = llm

    def create_search_agent(self) -> Any:
        """웹 검색 에이전트를 생성합니다."""
        brave_search = BraveSearch()
        return create_agent(self.llm, tools=[brave_search])

    def create_web_scraper_agent(self) -> Any:
        """웹 스크래핑 에이전트를 생성합니다."""
        return create_agent(
            self.llm,
            tools=[scrape_webpages],
            system_prompt=(
                "You are a web scraper agent. You will receive URLs from the previous search results. "
                "Use the scrape_webpages tool to extract detailed content from these URLs. "
                "Focus on extracting the most relevant information for the user's query."
            )
        )

    def create_doc_writer_agent(self) -> Any:
        """문서 작성 에이전트를 생성합니다."""
        return create_agent(
            self.llm,
            tools=[write_document, edit_document, read_document],
            system_prompt=(
                "You can read, write and edit documents. Start by creating a new document from the research results. "
                "Use write_document to create new files. If a document already exists, you can read and edit it. "
                "Don't ask follow-up questions - just complete the task."
            )
        )

    def create_note_taking_agent(self) -> Any:
        """노트 작성 에이전트를 생성합니다."""
        return create_agent(
            self.llm,
            tools=[create_outline, read_document],
            system_prompt=(
                "You can read documents and create outlines for the document writer. "
                "Don't ask follow-up questions."
            )
        )

    def create_chart_generating_agent(self) -> Any:
        """차트 생성 에이전트를 생성합니다."""
        return create_agent(
            self.llm,
            tools=[read_document, python_repl_tool]
        )


# =============================================================================
# SUPERVISOR FACTORY
# =============================================================================

def create_supervisor_node(
    llm: BaseChatModel,
    members: list[str],
    member_descriptions: Optional[dict[str, str]] = None,
    context: str = ""
) -> Callable[[State], Command[str]]:
    """
    Supervisor 노드를 생성하는 팩토리 함수

    Args:
        llm: 라우팅 결정에 사용할 언어 모델
        members: 관리할 워커 이름 목록
        member_descriptions: 워커별 설명 (선택)
        context: 추가 컨텍스트 (선택)

    Returns:
        Supervisor 노드 함수
    """
    # 워커 설명 구성
    member_info = ""
    if member_descriptions:
        member_info = "\n\nWorker capabilities:\n" + "\n".join(
            f"- {name}: {desc}" for name, desc in member_descriptions.items()
        )

    system_prompt = (
        f"You are a supervisor tasked with managing a conversation between the "
        f"following workers: {members}. Given the following user request, "
        f"respond with the worker to act next. Each worker will perform a "
        f"task and respond with their results and status.{member_info} "
        f"Analyze the conversation history carefully. If the user's request has been fully addressed and "
        f"sufficient information has been gathered, respond with FINISH. Only delegate additional work if "
        f"the current results are incomplete or the user's goals have not been met. {context}"
    )

    def supervisor_node(state: State) -> Command[str]:
        """LLM 기반 라우터"""
        messages = [
            {"role": "system", "content": system_prompt},
        ] + state["messages"]

        response = llm.with_structured_output(Router).invoke(messages)
        goto = response["next"]

        if goto == "FINISH":
            goto = END

        return Command(goto=goto, update={"next": goto})

    return supervisor_node


# =============================================================================
# RESEARCH GRAPH BUILDER
# =============================================================================

class ResearchGraphBuilder:
    """
    Research Graph를 구성하는 빌더 클래스

    구성 요소:
    - supervisor: reRank 기반 라우팅 결정
    - search: 웹 검색 (만족도 평가 및 재검색 포함)
    - web_scraper: URL 스크래핑 (사용자 제공 URL만)
    """

    def __init__(self, llm: BaseChatModel):
        self.llm = llm
        self.agent_factory = AgentFactory(llm)
        self.rerank_service = ReRankService(llm, config.relevance_threshold)

        # 에이전트 초기화
        self.search_agent = self.agent_factory.create_search_agent()
        self.web_scraper_agent = self.agent_factory.create_web_scraper_agent()

    def _create_search_node(self) -> Callable[[ResearchState], Command]:
        """검색 노드를 생성합니다."""

        def search_node(state: ResearchState) -> Command[Literal["supervisor"]]:
            """
            검색을 수행하고 reRank로 품질을 평가합니다.

            로직:
            1. 첫 번째 검색 수행
            2. 만족도 평가
            3. 50% 미만이면 질문 재구성 후 재검색
            4. 두 결과 중 더 좋은 것 선택
            5. search_agent 결과의 URL은 무시 (web_scraper 트리거 안 함)
            """
            print(f"\n{'='*60}")
            print(f"[SEARCH_NODE] 검색 시작...")

            user_message = state["messages"][0].content if state["messages"] else ""
            print(f"  - user_message: {user_message[:100]}...")

            # 첫 번째 검색
            print(f"  - BraveSearch API 호출 중...")
            result = self.search_agent.invoke(state)
            search_result = result["messages"][-1].content
            print(f"  - 검색 결과 수신 (길이: {len(search_result)})")

            first_score = self.rerank_service.evaluate_relevance(user_message, search_result)
            print(f"  - 첫 번째 관련성 점수: {first_score}")

            best_result = search_result
            best_score = first_score

            # 만족도가 임계값 미만이면 재검색
            if self.rerank_service.should_retry(first_score):
                print(f"  - 점수가 임계값({config.relevance_threshold}) 미만. 재검색 시도...")
                rephrased_query = self.rerank_service.rephrase_query(user_message, search_result)
                print(f"  - 재구성된 쿼리: {rephrased_query}")

                retry_state = {"messages": [HumanMessage(content=rephrased_query)]}
                print(f"  - 재검색 API 호출 중...")
                retry_result = self.search_agent.invoke(retry_state)
                retry_search_result = retry_result["messages"][-1].content

                second_score = self.rerank_service.evaluate_relevance(user_message, retry_search_result)
                print(f"  - 두 번째 관련성 점수: {second_score}")

                # 더 좋은 결과 선택
                if second_score > first_score:
                    best_result = retry_search_result
                    best_score = second_score
                    print(f"  - 두 번째 결과 선택 (더 높은 점수)")
                else:
                    print(f"  - 첫 번째 결과 유지")

            print(f"  [완료] 최종 점수: {best_score}")
            print(f"{'='*60}\n")

            return Command(
                update={
                    "messages": [HumanMessage(content=best_result, name="search")],
                    "urls_found": [],  # search_agent URL 무시
                    "search_completed": True,
                    "relevance_score": best_score,
                    "search_result": best_result,
                },
                goto="supervisor",
            )

        return search_node

    def _create_web_scraper_node(self) -> Callable[[ResearchState], Command]:
        """웹 스크래퍼 노드를 생성합니다."""

        def web_scraper_node(state: ResearchState) -> Command[Literal["supervisor"]]:
            """사용자가 제공한 URL을 스크래핑합니다."""
            print(f"\n{'='*60}")
            print(f"[WEB_SCRAPER_NODE] 스크래핑 시작...")

            urls = state.get("urls_found", [])
            print(f"  - urls_found: {urls}")

            if urls:
                print(f"  - URL이 제공됨. 스크래핑 진행...")
                url_message = f"Please scrape the following URLs to get detailed information: {urls[:config.max_scrape_urls]}"
                modified_state = {
                    "messages": state["messages"] + [HumanMessage(content=url_message)]
                }
                result = self.web_scraper_agent.invoke(modified_state)
                print(f"  - 스크래핑 완료 (결과 길이: {len(result['messages'][-1].content)})")
            else:
                print(f"  - URL이 없음. 기존 state로 진행...")
                result = self.web_scraper_agent.invoke(state)

            print(f"{'='*60}\n")

            return Command(
                update={
                    "messages": [
                        HumanMessage(content=result["messages"][-1].content, name="web_scraper")
                    ],
                    "scrape_completed": True,
                },
                goto="supervisor",
            )

        return web_scraper_node

    def _create_supervisor_node(self) -> Callable[[ResearchState], Command]:
        """Research 전용 Supervisor 노드를 생성합니다."""

        def research_supervisor_node(
            state: ResearchState
        ) -> Command[Literal["search", "web_scraper", "__end__"]]:
            """
            reRank 기반 라우팅을 수행합니다.

            라우팅 로직 (수정됨):
            1. 사용자 URL 제공 + 스크랩 미완료 → web_scraper (URL 있으면 검색 스킵)
            2. 검색 미완료 + URL 없음 → search
            3. 검색 완료 → END
            4. 스크랩 완료 → END
            """
            messages = state["messages"]
            search_completed = state.get("search_completed", False)
            scrape_completed = state.get("scrape_completed", False)
            relevance_score = state.get("relevance_score", 0.0)

            # 사용자 메시지에서 URL 추출
            user_message = messages[0].content if messages else ""
            user_provided_urls = extract_urls_from_text(user_message)

            # 디버그 로그 추가
            print(f"\n{'='*60}")
            print(f"[RESEARCH_SUPERVISOR] 라우팅 결정 중...")
            print(f"  - user_message: {user_message[:100]}...")
            print(f"  - user_provided_urls: {user_provided_urls}")
            print(f"  - search_completed: {search_completed}")
            print(f"  - scrape_completed: {scrape_completed}")
            print(f"  - relevance_score: {relevance_score}")

            # 라우팅 결정 (수정된 로직: URL 있으면 search 스킵)
            if user_provided_urls and not scrape_completed:
                # 사용자 제공 URL이 있으면 search 스킵하고 바로 web_scraper로
                goto = "web_scraper"
                print(f"  → URL이 제공됨. search 스킵, web_scraper로 이동")
            elif not search_completed and not user_provided_urls:
                # URL이 없고 검색도 안 됐으면 search 실행
                goto = "search"
                print(f"  → URL 없음. search 실행")
            elif scrape_completed or search_completed:
                # 스크랩이나 검색 완료되면 종료
                goto = END
                print(f"  → 작업 완료. END로 이동")
            else:
                goto = END
                print(f"  → 기본: END로 이동")

            print(f"  [결정] goto = {goto}")
            print(f"{'='*60}\n")

            # State 업데이트
            update_dict: dict[str, Any] = {
                "next": goto if goto != END else "FINISH"
            }

            if user_provided_urls:
                update_dict["urls_found"] = user_provided_urls

            return Command(goto=goto, update=update_dict)

        return research_supervisor_node

    def build(self) -> StateGraph:
        """Research Graph를 빌드하고 컴파일합니다."""
        builder = StateGraph(ResearchState)

        builder.add_node("supervisor", self._create_supervisor_node())
        builder.add_node("search", self._create_search_node())
        builder.add_node("web_scraper", self._create_web_scraper_node())

        builder.add_edge(START, "supervisor")

        return builder.compile()


# =============================================================================
# PAPER WRITING GRAPH BUILDER
# =============================================================================

class PaperWritingGraphBuilder:
    """
    Paper Writing Graph를 구성하는 빌더 클래스

    구성 요소:
    - supervisor: 작업 분배
    - doc_writer: 문서 작성/편집
    - note_taker: 아웃라인 생성
    - chart_generator: 차트 생성
    """

    def __init__(self, llm: BaseChatModel):
        self.llm = llm
        self.agent_factory = AgentFactory(llm)

        # 에이전트 초기화
        self.doc_writer_agent = self.agent_factory.create_doc_writer_agent()
        self.note_taking_agent = self.agent_factory.create_note_taking_agent()
        self.chart_generating_agent = self.agent_factory.create_chart_generating_agent()

    def _create_node(
        self,
        agent: Any,
        name: str
    ) -> Callable[[State], Command]:
        """공통 노드 생성 헬퍼"""

        def node(state: State) -> Command[Literal["supervisor"]]:
            result = agent.invoke(state)
            return Command(
                update={
                    "messages": [
                        HumanMessage(content=result["messages"][-1].content, name=name)
                    ]
                },
                goto="supervisor",
            )

        return node

    def build(self) -> StateGraph:
        """Paper Writing Graph를 빌드하고 컴파일합니다."""
        # Supervisor 생성
        supervisor_node = create_supervisor_node(
            self.llm,
            ["doc_writer", "note_taker", "chart_generator"],
            {
                "doc_writer": "Writes, edits, and reads documents. Use this to create or modify the final document content.",
                "note_taker": "Creates outlines and reads documents. Use this to organize information and create structured outlines before writing.",
                "chart_generator": "Reads documents and generates charts using Python code. Use this when visualizations or data charts are needed."
            },
            context="Start with doc_writer to create a new document from the research results. Then FINISH."
        )

        # 그래프 빌드
        builder = StateGraph(State)

        builder.add_node("supervisor", supervisor_node)
        builder.add_node("doc_writer", self._create_node(self.doc_writer_agent, "doc_writer"))
        builder.add_node("note_taker", self._create_node(self.note_taking_agent, "note_taker"))
        builder.add_node("chart_generator", self._create_node(self.chart_generating_agent, "chart_generator"))

        builder.add_edge(START, "supervisor")

        return builder.compile()


# =============================================================================
# SUPER GRAPH BUILDER
# =============================================================================

class SuperGraphBuilder:
    """
    Super Graph (최상위 그래프)를 구성하는 빌더 클래스

    구성 요소:
    - supervisor: 팀 간 작업 조율
    - research_team: Research Graph 호출
    - writing_team: Paper Writing Graph 호출
    """

    def __init__(self, llm: BaseChatModel):
        self.llm = llm

        # 하위 그래프 빌드
        self.research_graph = ResearchGraphBuilder(llm).build()
        self.paper_writing_graph = PaperWritingGraphBuilder(llm).build()

    def _create_research_team_node(self) -> Callable[[State], Command]:
        """Research Team 노드를 생성합니다."""

        def call_research_team(state: State) -> Command[Literal["supervisor"]]:
            """Research Graph를 호출합니다."""
            print(f"\n{'#'*60}")
            print(f"[RESEARCH_TEAM] Research Graph 시작")
            print(f"  - messages 수: {len(state['messages'])}")
            if state['messages']:
                user_msg = state['messages'][0]
                content = user_msg.content if hasattr(user_msg, 'content') else str(user_msg)
                print(f"  - 첫 메시지: {content[:100]}...")
            print(f"{'#'*60}\n")

            response = self.research_graph.invoke({
                "messages": state["messages"],
                "urls_found": [],
                "search_completed": False,
                "scrape_completed": False,
                "relevance_score": 0.0,
                "search_result": "",
            })

            print(f"\n{'#'*60}")
            print(f"[RESEARCH_TEAM] Research Graph 완료")
            print(f"  - 응답 메시지 수: {len(response['messages'])}")
            print(f"{'#'*60}\n")

            return Command(
                update={
                    "messages": [
                        HumanMessage(
                            content=response["messages"][-1].content,
                            name="research_team"
                        )
                    ]
                },
                goto="supervisor",
            )

        return call_research_team

    def _create_writing_team_node(self) -> Callable[[State], Command]:
        """Writing Team 노드를 생성합니다."""

        def call_paper_writing_team(state: State) -> Command[Literal["supervisor"]]:
            """Paper Writing Graph를 호출합니다."""
            response = self.paper_writing_graph.invoke(state)

            return Command(
                update={
                    "messages": [
                        HumanMessage(
                            content=response["messages"][-1].content,
                            name="writing_team"
                        )
                    ]
                },
                goto="supervisor",
            )

        return call_paper_writing_team

    def build(self) -> StateGraph:
        """Super Graph를 빌드하고 컴파일합니다."""
        # Supervisor 생성
        supervisor_node = create_supervisor_node(
            self.llm,
            ["research_team", "writing_team"],
            {
                "research_team": "Searches the web and scrapes webpages to gather information. Use this when you need to research a topic or collect data.",
                "writing_team": "Creates outlines, writes documents, and generates charts. Use this when you need to produce written content or visualizations."
            }
        )

        # 그래프 빌드
        builder = StateGraph(State)

        builder.add_node("supervisor", supervisor_node)
        builder.add_node("research_team", self._create_research_team_node())
        builder.add_node("writing_team", self._create_writing_team_node())

        builder.add_edge(START, "supervisor")

        return builder.compile()


# =============================================================================
# MAIN APPLICATION
# =============================================================================

class HierarchicalAgentTeams:
    """
    계층적 에이전트 팀 시스템의 메인 클래스

    사용 예:
        app = HierarchicalAgentTeams()
        result = app.run("독도에 관한 글을 작성하세요.")
    """

    def __init__(self, model_name: str = None):
        """
        Args:
            model_name: 사용할 LLM 모델명 (기본값: config.model_name)
        """
        self.model_name = model_name or config.model_name
        self.llm = ChatOpenAI(model=self.model_name)
        self.graph = SuperGraphBuilder(self.llm).build()

    def run(self, user_request: str, stream: bool = True) -> Any:
        """
        사용자 요청을 처리합니다.

        Args:
            user_request: 사용자의 요청 메시지
            stream: 스트리밍 모드 사용 여부

        Returns:
            처리 결과
        """
        input_data = {
            "messages": [("user", user_request)]
        }

        if stream:
            return self._run_stream(input_data)
        else:
            return self._run_invoke(input_data)

    def _run_stream(self, input_data: dict) -> None:
        """스트리밍 모드로 실행합니다."""
        for state in self.graph.stream(
            input_data,
            {"recursion_limit": config.recursion_limit}
        ):
            print(state)
            print("---")

    def _run_invoke(self, input_data: dict) -> dict:
        """일반 모드로 실행합니다."""
        return self.graph.invoke(
            input_data,
            {"recursion_limit": config.recursion_limit}
        )


# =============================================================================
# ENTRY POINT
# =============================================================================

def main():
    """메인 실행 함수"""
    app = HierarchicalAgentTeams()

    # 예시 요청 실행
    user_request = (
        "https://www.aitimes.com/news/articleView.html?idxno=206185 가사를 읽고 "
        "기사 내용을 바탕으로 1500자 내외로 블로그를 작성하세요."
    )

    print("=" * 80)
    print("Hierarchical Agent Teams - Starting")
    print("=" * 80)
    print(f"User Request: {user_request}")
    print("=" * 80)

    app.run(user_request, stream=True)


if __name__ == "__main__":
    main()
