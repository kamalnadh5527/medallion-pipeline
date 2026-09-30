from typing import Any

import chromadb

from core.config import CHROMA_DIR


COLLECTION_NAME = "idamp_memory"


def get_chroma_client() -> chromadb.PersistentClient:
    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(path=str(CHROMA_DIR))


def get_collection(name: str = COLLECTION_NAME) -> Any:
    return get_chroma_client().get_or_create_collection(
        name=name,
        embedding_function=None,
    )


def store_document(
    doc_id: str,
    text: str,
    metadata: dict[str, Any] | None = None,
) -> None:
    try:
        collection = get_collection()
        options: dict[str, Any] = {
            "ids": [doc_id],
            "documents": [text],
            # Chroma requires vectors; this placeholder is not a semantic embedding.
            "embeddings": [[0.0]],
        }
        if metadata is not None:
            options["metadatas"] = [metadata]

        collection.upsert(**options)
    except Exception:
        pass


def query_memory(
    query_text: str,
    n_results: int = 5,
) -> list[dict[str, Any]]:
    try:
        collection = get_collection()
        results = collection.get(
            limit=n_results,
            include=["documents", "metadatas"],
        )

        ids = results.get("ids", [])
        documents = results.get("documents") or []
        metadatas = results.get("metadatas") or []

        return [
            {
                "id": doc_id,
                "document": documents[index] if index < len(documents) else None,
                "metadata": metadatas[index] if index < len(metadatas) else None,
            }
            for index, doc_id in enumerate(ids)
        ]
    except Exception:
        return []