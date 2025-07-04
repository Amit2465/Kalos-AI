from typing import Dict
import chromadb
from chromadb.utils import embedding_functions
from chromadb.api.models.Collection import Collection
import logging


def init_chroma_vector_store(
    collection_name: str = "injection_detection",
    persist_directory: str = "./chroma_storage"
) -> Collection:
    """
    Initializes and returns a ChromaDB collection with sentence transformer embeddings.

    Args:
        collection_name (str): Name of the ChromaDB collection.
        persist_directory (str): Path to persist ChromaDB data.

    Returns:
        Collection: A ChromaDB collection with embeddings.
    """
    try:
        client = chromadb.PersistentClient(path=persist_directory)

        embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="all-MiniLM-L6-v2"
        )

        existing_collections = [col.name for col in client.list_collections()]

        if collection_name not in existing_collections:
            return client.create_collection(name=collection_name, embedding_function=embedding_fn)
        else:
            return client.get_collection(name=collection_name, embedding_function=embedding_fn)
    except Exception as e:
        logging.error(f"Failed to initialize Chroma vector store: {e}")
        return None


def detect_pi_using_vector_database(
    user_input: str,
    similarity_threshold: float,
    collection: Collection
) -> Dict:
    """
    Detects prompt injection by similarity with known injection examples.

    Args:
        user_input (str): Input prompt from user.
        similarity_threshold (float): Threshold above which similarity is flagged.
        collection (Collection): Chroma collection with known injection embeddings.

    Returns:
        Dict: Dictionary with top similarity score and match count above threshold.
    """
    try:
        top_k = 10
        results = collection.query(query_texts=[user_input], n_results=top_k)

        distances = results.get("distances", [[]])[0]
        top_score = 0.0
        count_over_threshold = 0

        for distance in distances:
            similarity = 1 - distance 
            top_score = max(top_score, similarity)
            if similarity >= similarity_threshold:
                count_over_threshold += 1

        return {
            "top_score": round(top_score, 4),
            "count_over_max_vector_score": count_over_threshold
        }
    except Exception as e:
        logging.error(f"Vector database detection failed: {e}")
        return {"top_score": 0.0, "count_over_max_vector_score": 0}
