import chromadb
from sentence_transformers import SentenceTransformer

client = chromadb.PersistentClient(path="chroma_db")
collection = client.get_or_create_collection("buywisely_products")

embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

def add_product_data(product_id: str, texts: list):
    embeddings = embedding_model.encode(texts).tolist()

    collection.add(
        ids=[f"{product_id}_{i}" for i in range(len(texts))],
        documents=texts,
        embeddings=embeddings,
        metadatas=[{"product_id": product_id} for _ in texts]
    )

def retrieve_context(question: str, n_results: int = 3):
    query_embedding = embedding_model.encode([question]).tolist()

    results = collection.query(
        query_embeddings=query_embedding,
        n_results=n_results
    )

    return results["documents"][0]