from services.knowledge.rag_service import KnowledgeService

def test_knowledge_service_add_document_and_search_returns_stored_entries():
    rag = KnowledgeService()

    rag.add_document("AI platforms are powerful")
    rag.add_document("Second document", metadata={"source": "doc-2"})

    res = rag.search("AI")

    assert len(res) == 2
    assert res[0]["text"] == "AI platforms are powerful"
    assert res[0]["vector"] == [25.0]
    assert res[1]["metadata"] == {"source": "doc-2"}
