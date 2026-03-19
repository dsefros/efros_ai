from services.knowledge.rag_service import KnowledgeService

def test_rag():

    rag = KnowledgeService()

    rag.add_document("AI platforms are powerful")

    res = rag.search("AI")

    print("RAG RESULT:", res)

if __name__ == "__main__":
    test_rag()
