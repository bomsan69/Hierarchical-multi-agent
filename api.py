"""
FastAPI REST API Server for Hierarchical Agent Teams

이 모듈은 main.py의 HierarchicalAgentTeams를 REST API로 제공합니다.

Endpoints:
    POST /make_report - 리포트 주제를 받아 리포트 생성
"""

from __future__ import annotations

import logging
from typing import Any

from fastapi import FastAPI, HTTPException, Security, Depends
from fastapi.security import APIKeyHeader
from pydantic import BaseModel, Field

from main import HierarchicalAgentTeams, agent_logger

# API 키 설정
API_KEY = "5jang12345678##*"
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=True)

# FastAPI 앱 초기화
app = FastAPI(
    title="Hierarchical Agent Teams API",
    description="REST API for generating reports using hierarchical agent teams",
    version="1.0.0"
)


# =============================================================================
# AUTHENTICATION
# =============================================================================

async def verify_api_key(api_key: str = Security(api_key_header)) -> str:
    """
    API 키를 검증합니다.
    
    Args:
        api_key: Header에서 받은 API 키
        
    Returns:
        검증된 API 키
        
    Raises:
        HTTPException: API 키가 유효하지 않을 때
    """
    if api_key != API_KEY:
        agent_logger.warning(f"[API] 유효하지 않은 API 키 시도: {api_key[:10]}...")
        raise HTTPException(
            status_code=401,
            detail="Invalid API Key"
        )
    return api_key


# =============================================================================
# REQUEST/RESPONSE MODELS
# =============================================================================

class MakeReportRequest(BaseModel):
    """리포트 생성 요청 스키마"""
    topic: str = Field(
        ...,
        description="리포트 주제 또는 작성 요청",
        min_length=1,
        examples=["독도에 관한 글을 작성하세요."]
    )


class MakeReportResponse(BaseModel):
    """리포트 생성 응답 스키마"""
    success: bool = Field(description="성공 여부")
    topic: str = Field(description="요청한 리포트 주제")
    result: dict[str, Any] = Field(description="생성된 리포트 결과")
    message: str = Field(description="처리 결과 메시지")


# =============================================================================
# ENDPOINTS
# =============================================================================

@app.post("/make_report", response_model=MakeReportResponse)
async def make_report(
    request: MakeReportRequest,
    api_key: str = Depends(verify_api_key)
) -> MakeReportResponse:
    """
    리포트를 생성합니다.

    Args:
        request: 리포트 주제를 포함한 요청

    Returns:
        생성된 리포트 결과

    Raises:
        HTTPException: 리포트 생성 실패 시
    """
    agent_logger.info("=" * 60)
    agent_logger.info(f"[API] POST /make_report 요청 수신")
    agent_logger.info(f"[API] Topic: {request.topic}")
    agent_logger.info("=" * 60)

    try:
        # HierarchicalAgentTeams 초기화
        agent_teams = HierarchicalAgentTeams()

        # 리포트 생성 (stream=False로 결과 받기)
        result = agent_teams.run(request.topic, stream=False)

        agent_logger.info("[API] 리포트 생성 성공")
        agent_logger.info("=" * 60)

        # 응답 생성
        return MakeReportResponse(
            success=True,
            topic=request.topic,
            result=result,
            message="Report generated successfully"
        )

    except Exception as e:
        agent_logger.error(f"[API] 리포트 생성 실패: {str(e)}", exc_info=True)
        agent_logger.info("=" * 60)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to generate report: {str(e)}"
        )


@app.get("/")
async def root():
    """루트 엔드포인트 - API 정보 반환"""
    return {
        "name": "Hierarchical Agent Teams API",
        "version": "1.0.0",
        "endpoints": {
            "POST /make_report": "Generate a report based on the given topic"
        }
    }


@app.get("/health")
async def health_check():
    """헬스 체크 엔드포인트"""
    return {"status": "ok"}


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    import uvicorn

    agent_logger.info("Starting FastAPI server...")
    uvicorn.run(
        "api:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
