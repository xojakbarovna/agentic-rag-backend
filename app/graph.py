import os
from typing import List, TypedDict, Type
from pydantic import BaseModel, Field
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_openai import ChatOpenAI
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams
from langgraph.graph import StateGraph, END
from sentence_transformers import SentenceTransformer
from ddgs import DDGS

# --- Config (all overridable via environment variables) ---
DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY")
if not DEEPSEEK_API_KEY:
    raise RuntimeError(
        "DEEPSEEK_API_KEY environment variable is not set. "
        "Set it in your environment (never hardcode it in source)."
    )

DEEPSEEK_MODEL = os.environ.get("DEEPSEEK_MODEL", "deepseek-chat")
DEEPSEEK_BASE_URL = os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1")

EMBEDDING_MODEL_NAME = os.environ.get("EMBEDDING_MODEL_NAME", "BAAI/bge-small-en-v1.5")
EMBEDDING_DIM = 384  # bge-small-en-v1.5 output dimension

QDRANT_PATH = os.environ.get("QDRANT_PATH", "/tmp/qdrant_embedded")
MAX_RETRIES = int(os.environ.get("MAX_RETRIES", "2"))


class LocalEmbeddings(Embeddings):
    """Wraps a sentence-transformers model so it satisfies LangChain's Embeddings interface
    (embed_documents / embed_query), which plain SentenceTransformer does not implement."""

    def __init__(self, model_name: str = EMBEDDING_MODEL_NAME):
        self._model = SentenceTransformer(model_name)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self._model.encode(texts, convert_to_numpy=True).tolist()

    def embed_query(self, text: str) -> List[float]:
        return self._model.encode([text], convert_to_numpy=True)[0].tolist()


class LLMClient:
    def __init__(self):
        self.llm = ChatOpenAI(
            model=DEEPSEEK_MODEL,
            temperature=0,
            api_key=DEEPSEEK_API_KEY,
            base_url=DEEPSEEK_BASE_URL,
        )
        self.embeddings = LocalEmbeddings()

    def get_structured_grader(self, schema: Type[BaseModel]):
        return self.llm.with_structured_output(schema)


client_wrapper = LLMClient()

qdrant_client = QdrantClient(path=QDRANT_PATH)
collection_name = "agentic_rag_collection"

if not qdrant_client.collection_exists(collection_name):
    qdrant_client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=EMBEDDING_DIM, distance=Distance.COSINE),
    )

vector_store = QdrantVectorStore(
    client=qdrant_client,
    collection_name=collection_name,
    embedding=client_wrapper.embeddings,
)

retriever = vector_store.as_retriever(search_kwargs={"k": 4})


class GraphState(TypedDict):
    question: str
    documents: List[Document]
    web_results: str
    generation: str
    steps: List[str]
    retry_count: int


class GradeDocuments(BaseModel):
    relevant: bool = Field(description="Is the document relevant to the user question?")


class GradeGeneration(BaseModel):
    grounded: bool = Field(description="Is the answer completely supported by the provided facts?")
    useful: bool = Field(description="Does the answer directly resolve the user's question?")


def retrieve_node(state: GraphState):
    docs = retriever.invoke(state["question"])
    return {
        "documents": docs,
        "steps": state.get("steps", []) + ["retrieve"],
        "retry_count": state.get("retry_count", 0),
    }


def grade_documents_node(state: GraphState):
    grader = client_wrapper.get_structured_grader(GradeDocuments)
    relevant_docs = []
    for doc in state["documents"]:
        prompt = (
            f"Document:\n{doc.page_content}\n\n"
            f"Question: {state['question']}\nIs this document relevant? Answer yes or no."
        )
        res = grader.invoke(prompt)
        if res.relevant:
            relevant_docs.append(doc)
    return {"documents": relevant_docs, "steps": state["steps"] + ["grade_documents"]}


def web_search_node(state: GraphState):
    """Real web search fallback (previously a hardcoded mock string)."""
    query = state["question"]
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=3))
        if results:
            combined = "\n".join(
                f"- {r.get('title', '')}: {r.get('body', '')}" for r in results
            )
        else:
            combined = "No web results found."
    except Exception as e:
        combined = f"Web search failed: {e}"
    return {"web_results": combined, "steps": state["steps"] + ["web_search"]}


def generate_node(state: GraphState):
    context = "\n".join([d.page_content for d in state["documents"]])
    if state.get("web_results"):
        context += f"\nWeb Context: {state['web_results']}"
    prompt = f"Answer the question: {state['question']}\nUsing only this context:\n{context}"
    res = client_wrapper.llm.invoke(prompt)
    return {
        "generation": res.content,
        "steps": state["steps"] + ["generate"],
        # Increment on every generation attempt so route_after_generate's retry
        # limit can actually be reached (previously this counter never advanced,
        # allowing an infinite generate/web_search loop).
        "retry_count": state.get("retry_count", 0) + 1,
    }


def route_after_grade(state: GraphState):
    if not state["documents"]:
        return "web_search"
    return "generate"


def route_after_generate(state: GraphState):
    if state.get("retry_count", 0) >= MAX_RETRIES:
        return "useful"
    grader = client_wrapper.get_structured_grader(GradeGeneration)
    context = "\n".join([d.page_content for d in state["documents"]])
    prompt = f"Context: {context}\nGeneration: {state['generation']}\nQuestion: {state['question']}"
    res = grader.invoke(prompt)
    if not res.grounded:
        return "not_grounded"
    if not res.useful:
        return "not_useful"
    return "useful"


workflow = StateGraph(GraphState)
workflow.add_node("retrieve", retrieve_node)
workflow.add_node("grade_documents", grade_documents_node)
workflow.add_node("web_search", web_search_node)
workflow.add_node("generate", generate_node)

workflow.set_entry_point("retrieve")
workflow.add_edge("retrieve", "grade_documents")
workflow.add_conditional_edges(
    "grade_documents", route_after_grade, {"web_search": "web_search", "generate": "generate"}
)
workflow.add_edge("web_search", "generate")
workflow.add_conditional_edges(
    "generate",
    route_after_generate,
    {"useful": END, "not_grounded": "generate", "not_useful": "web_search"},
)

rag_agent = workflow.compile()
