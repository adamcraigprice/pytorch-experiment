"""
Tests for the AI Productivity Assistant
========================================
Covers each module (summarizer, action_items, prioritizer, vector_search)
and the FastAPI endpoints.

Run with:
    pytest tests/ -v
"""

import pytest
from fastapi.testclient import TestClient

# ---------------------------------------------------------------------------
# Sample transcript used across tests
# ---------------------------------------------------------------------------

SAMPLE_TRANSCRIPT = """
Meeting: Q1 Sprint Planning
Date: 2026-03-01
Attendees: Alice, Bob, Carol, Dave

Alice: Welcome everyone. Let's review the sprint goals.
Bob: I've finished the authentication module. We still need to integrate it.
Carol: Action item: Carol will update the API documentation by Friday.
Dave: TODO: Dave needs to fix the CI pipeline — it's been failing since last week.
Alice: Bob should deploy the auth module to staging by Wednesday.
Carol: Please review the pull requests before end of day.
Alice: Make sure everyone updates their Jira tickets.
Bob: The database migration is a blocker — we need to resolve it ASAP.
Dave: I'll set up monitoring dashboards by next week.
Alice: Follow-up: schedule a design review for the new dashboard feature.
Carol: Reminder — the client demo is tomorrow, we need the staging environment ready today.
Alice: Assigned to Bob: prepare the demo script by end of day.
Dave: Task: investigate the memory leak in the worker service.
Alice: That's all for today. Thanks everyone.
"""

SAMPLE_SHORT = "Bob will fix the login bug by Friday. TODO: update the docs."


# ===========================================================================
# Unit Tests – Summarizer
# ===========================================================================

class TestSummarizer:
    def test_summarize_returns_string(self):
        from backend.summarizer import summarize_text
        result = summarize_text(SAMPLE_SHORT)
        assert isinstance(result, str)
        assert len(result) > 0

    def test_summarize_empty(self):
        from backend.summarizer import summarize_text
        assert summarize_text("") == ""
        assert summarize_text("   ") == ""

    def test_summarize_long_text(self):
        from backend.summarizer import summarize_text
        result = summarize_text(SAMPLE_TRANSCRIPT)
        assert isinstance(result, str)
        assert len(result) > 10


# ===========================================================================
# Unit Tests – Action Items
# ===========================================================================

class TestActionItems:
    def test_extract_finds_items(self):
        from backend.action_items import extract_action_items
        items = extract_action_items(SAMPLE_TRANSCRIPT)
        assert isinstance(items, list)
        assert len(items) > 0
        # Each item should be a dict with at least a "text" key
        for item in items:
            assert "text" in item
            assert "confidence" in item

    def test_extract_empty(self):
        from backend.action_items import extract_action_items
        assert extract_action_items("") == []

    def test_extract_assignee_detection(self):
        from backend.action_items import extract_action_items
        items = extract_action_items("Alice will prepare the report by Friday.")
        assert len(items) >= 1

    def test_extract_todo_keyword(self):
        from backend.action_items import extract_action_items
        items = extract_action_items("TODO: Refactor the codebase.")
        assert len(items) >= 1
        assert any("refactor" in item["text"].lower() for item in items)


# ===========================================================================
# Unit Tests – Prioritizer
# ===========================================================================

class TestPrioritizer:
    def test_prioritize_returns_list(self):
        from backend.prioritizer import prioritize_tasks
        tasks = [
            {"text": "Fix the critical production bug ASAP"},
            {"text": "Update the README file"},
            {"text": "Urgent: deploy hotfix immediately"},
        ]
        result = prioritize_tasks(tasks)
        assert isinstance(result, list)
        assert len(result) == 3

    def test_prioritize_scoring(self):
        from backend.prioritizer import prioritize_tasks
        tasks = [
            {"text": "Update docs when convenient"},
            {"text": "URGENT critical blocker ASAP — fix immediately"},
        ]
        result = prioritize_tasks(tasks)
        # The urgent task should be ranked first
        assert result[0]["priority_score"] >= result[1]["priority_score"]
        assert result[0]["priority_label"] in ("high", "medium")

    def test_prioritize_empty(self):
        from backend.prioritizer import prioritize_tasks
        assert prioritize_tasks([]) == []

    def test_prioritize_string_input(self):
        from backend.prioritizer import prioritize_tasks
        result = prioritize_tasks(["Fix login bug", "Update docs"])
        assert len(result) == 2
        for t in result:
            assert "priority_score" in t
            assert "priority_label" in t


# ===========================================================================
# Unit Tests – Vector Search
# ===========================================================================

class TestVectorSearch:
    def test_add_and_search(self):
        from backend.vector_search import VectorStore
        store = VectorStore()
        store.clear()

        store.add_text("The authentication module failed during deployment.")
        store.add_text("We discussed the quarterly earnings report.")
        store.add_text("The CI pipeline is broken and needs fixing.")

        results = store.search("deployment authentication", k=2)
        assert len(results) > 0
        assert results[0]["score"] > 0  # should have positive similarity

    def test_search_empty_store(self):
        from backend.vector_search import VectorStore
        store = VectorStore()
        store.clear()
        results = store.search("anything")
        assert results == []

    def test_add_batch(self):
        from backend.vector_search import VectorStore
        store = VectorStore()
        store.clear()

        ids = store.add_documents(
            ["Document one about ML.", "Document two about DevOps."],
            source="test",
        )
        assert len(ids) == 2
        assert store.total_documents == 2


# ===========================================================================
# Integration Tests – FastAPI Endpoints
# ===========================================================================

class TestAPI:
    @pytest.fixture(autouse=True)
    def client(self):
        from backend.main import app
        self.client = TestClient(app)

    def test_health(self):
        resp = self.client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "healthy"

    def test_summarize_json(self):
        resp = self.client.post("/summarize", json={"text": SAMPLE_SHORT})
        assert resp.status_code == 200
        assert "summary" in resp.json()

    def test_action_items_json(self):
        resp = self.client.post("/action-items", json={"text": SAMPLE_TRANSCRIPT})
        assert resp.status_code == 200
        data = resp.json()
        assert "action_items" in data
        assert data["count"] > 0

    def test_prioritize_json(self):
        resp = self.client.post("/prioritize", json={"text": SAMPLE_TRANSCRIPT})
        assert resp.status_code == 200
        data = resp.json()
        assert "prioritized_tasks" in data

    def test_full_analysis(self):
        resp = self.client.post("/analyze", json={"text": SAMPLE_TRANSCRIPT})
        assert resp.status_code == 200
        data = resp.json()
        assert "summary" in data
        assert "action_items" in data
        assert "prioritized_tasks" in data

    def test_index_and_search(self):
        # Index
        resp = self.client.post("/index", json={
            "text": "Discussion about PyTorch model training.",
            "source": "test",
        })
        assert resp.status_code == 200

        # Search
        resp = self.client.post("/search", json={
            "query": "PyTorch training",
            "k": 3,
        })
        assert resp.status_code == 200
        assert "results" in resp.json()

    def test_empty_input_rejected(self):
        resp = self.client.post("/summarize", json={"text": ""})
        assert resp.status_code == 422  # pydantic min_length validation
