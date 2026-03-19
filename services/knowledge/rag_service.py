class MockEmbedder:

    def embed(self, text):
        return [float(len(text))]

class MockVectorStore:

    def __init__(self):
        self.data = []

    def add(self, item):
        self.data.append(item)

    def search(self, vector, k):

        return self.data[:k]

class KnowledgeService:

    def __init__(self, embedder=None, vector_store=None):

        self.embedder = embedder or MockEmbedder()
        self.vector_store = vector_store or MockVectorStore()

    def add_document(self, text, metadata=None):

        emb = self.embedder.embed(text)

        self.vector_store.add({
            "vector": emb,
            "text": text,
            "metadata": metadata
        })

    def search(self, query, k=3):

        q = self.embedder.embed(query)

        return self.vector_store.search(q, k)
