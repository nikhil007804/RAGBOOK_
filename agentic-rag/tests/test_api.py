"""Tests for FastAPI routes."""

from fastapi.testclient import TestClient

from api.main import app
from api import router as router_module


class DummyCache:
    """In-memory cache stub."""

    def __init__(self) -> None:
        self.data = {}
        self.client = self

    def ping(self) -> bool:
        """Return healthy status."""
        return True

    def check_cache(self, key: str):
        """Lookup key in local dict."""
        return self.data.get(key)

    def save_cache(self, key: str, payload):
        """Save key to local dict."""
        self.data[key] = payload


class DummyEmbedder:
    """Dummy query embedder."""

    def embed(self, query: str):
        """Return fixed dense vector."""
        return [0.1] * 1024


class DummyRetriever:
    """Dummy retriever."""

    def retrieve(self, query_text: str, dense_vector, top_k: int, alpha=None):
        """Return deterministic fake matches."""
        match = {
            "id": "ch1_p10_prose_1",
            "metadata": {
                "text": "Agent loops coordinate planning and execution.",
                "page_start": 10,
                "chapter_title": "Agent Loops",
                "content_type": "prose",
            },
        }
        return [match], [match], [match], "conceptual", 0.8


class DummyReranker:
    """Dummy reranker."""

    def rerank(self, query: str, results, top_n: int = 5):
        """Return the first top_n results."""
        return results[:top_n]


class DummyLLM:
    """Dummy llm client."""

    def generate(self, prompt: str) -> str:
        """Return fixed JSON output."""
        return '{"answer":"Agent loops iterate until stop conditions.","citations":[{"page":10,"chapter":"Agent Loops","excerpt":"iterate until stop condition"}],"confidence":"high","query_type":"prose"}'


def test_query_endpoint_pipeline(monkeypatch) -> None:
    """POST /query should return pipeline payload on cache miss."""
    dummy_cache = DummyCache()
    def mock_components():
        return {
            "cache_manager": dummy_cache,
            "query_embedder": DummyEmbedder(),
            "retriever": DummyRetriever(),
            "reranker": DummyReranker(),
            "llm_client": DummyLLM(),
        }
    mock_components.cache_clear = lambda: None

    monkeypatch.setattr(router_module, "_components", mock_components)
    client = TestClient(app)
    response = client.post("/query", json={"query": "What is an agent loop?"})
    assert response.status_code == 200
    body = response.json()
    assert body["source"] == "pipeline"
    assert body["confidence"] == "high"


def test_health_endpoint(monkeypatch) -> None:
    """GET /health should return status payload."""
    def mock_components():
        return {
            "cache_manager": DummyCache(),
            "query_embedder": DummyEmbedder(),
            "retriever": DummyRetriever(),
            "reranker": DummyReranker(),
            "llm_client": DummyLLM(),
        }
    mock_components.cache_clear = lambda: None

    monkeypatch.setattr(router_module, "_components", mock_components)
    client = TestClient(app)
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"
