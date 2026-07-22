import os
from typing import List

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.graph import rag_agent, vector_store

app = FastAPI(title="Agentic RAG Engine API", version="1.0.0")

# CORS: allow_origins=["*"] + allow_credentials=True is invalid per the CORS spec
# (browsers reject it) and overly permissive besides. Default to no credentials
# unless the deployer explicitly sets ALLOWED_ORIGINS to a real origin list.
_allowed_origins_env = os.environ.get("ALLOWED_ORIGINS", "")
ALLOWED_ORIGINS = [o.strip() for o in _allowed_origins_env.split(",") if o.strip()]

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS if ALLOWED_ORIGINS else ["*"],
    allow_credentials=bool(ALLOWED_ORIGINS),
    allow_methods=["*"],
    allow_headers=["*"],
)

text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)


class QueryRequest(BaseModel):
    question: str


class QueryResponse(BaseModel):
    answer: str
    steps: List[str]


class IngestRequest(BaseModel):
    documents: List[str]


class IngestResponse(BaseModel):
    chunks_added: int


@app.get("/health")
def health_check():
    return {"status": "healthy"}


@app.post("/ingest", response_model=IngestResponse)
def ingest_endpoint(payload: IngestRequest):
    """Previously missing entirely: there was no way to populate the vector
    store, so /chat always retrieved from an empty collection."""
    if not payload.documents:
        raise HTTPException(status_code=400, detail="documents list cannot be empty")
    try:
        chunks: List[Document] = []
        for text in payload.documents:
            for chunk_text in text_splitter.split_text(text):
                chunks.append(Document(page_content=chunk_text))
        if chunks:
            vector_store.add_documents(chunks)
        return IngestResponse(chunks_added=len(chunks))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/chat", response_model=QueryResponse)
def chat_endpoint(payload: QueryRequest):
    try:
        inputs = {"question": payload.question, "retry_count": 0, "steps": []}
        result = rag_agent.invoke(inputs)
        return QueryResponse(
            answer=result["generation"],
            steps=result["steps"],
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
