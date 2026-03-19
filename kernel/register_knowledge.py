from services.knowledge.rag_engine import KnowledgeEngine


def register_knowledge(kernel):
    engine = KnowledgeEngine(model_manager=kernel.model_manager)
    kernel.knowledge = engine
