from services.knowledge.rag_service import KnowledgeService


def test_rag_add_document_and_search_returns_stored_entries():
    rag = KnowledgeService()
    rag.add_document("AI platforms are powerful", metadata={"source": "doc-1"})
    rag.add_document("AI copilots help operators", metadata={"source": "doc-2"})

    result = rag.search("AI", k=2)

    assert len(result) == 2
    assert result[0]["text"] == "AI platforms are powerful"
    assert result[0]["metadata"] == {"source": "doc-1"}
    assert result[1]["text"] == "AI copilots help operators"


def test_rag_search_respects_requested_limit():
    rag = KnowledgeService()
    for idx in range(4):
        rag.add_document(f"doc {idx}")

    result = rag.search("doc", k=3)

    assert len(result) == 3
