from fastapi import APIRouter

router = APIRouter()

def register_rag_routes(app, kernel):

    @app.post("/rag/answer")
    def rag_answer(payload: dict):

        result = kernel.knowledge.answer(payload["query"])

        return result
