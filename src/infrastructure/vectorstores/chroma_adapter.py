from pathlib import Path

import chromadb

from src.domain.models import DocumentChunk
from src.domain.ports import VectorStorePort


class ChromaVectorStoreAdapter(VectorStorePort):
    """Adaptador de ChromaDB persistente en disco con distancia coseno."""

    def __init__(self, persist_dir: Path, collection_name: str = "rag_conductor") -> None:
        persist_dir.mkdir(parents=True, exist_ok=True)
        self._client = chromadb.PersistentClient(path=str(persist_dir))
        self._collection_name = collection_name
        self._collection = self._get_or_create_collection()

    def _get_or_create_collection(self):
        return self._client.get_or_create_collection(
            name=self._collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def upsert(self, chunks: list[DocumentChunk], embeddings: list[list[float]]) -> int:
        if not chunks:
            return 0
        if len(chunks) != len(embeddings):
            raise ValueError("La cantidad de chunks y embeddings debe coincidir.")

        self._collection.upsert(
            ids=[chunk.id for chunk in chunks],
            documents=[chunk.content for chunk in chunks],
            embeddings=embeddings,
            metadatas=[
                {
                    "source": chunk.source,
                    "page": chunk.page,
                    "file_hash": chunk.file_hash,
                }
                for chunk in chunks
            ],
        )
        return len(chunks)

    def search(
        self,
        query_embedding: list[float],
        top_k: int = 5,
        min_score: float = 0.0,
    ) -> list[DocumentChunk]:
        total = self.count()
        if total == 0:
            return []

        results = self._collection.query(
            query_embeddings=[query_embedding],
            n_results=min(top_k, total),
            include=["documents", "metadatas", "distances"],
        )

        ids = results.get("ids", [[]])[0]
        documents = results.get("documents", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]
        distances = results.get("distances", [[]])[0]

        matched: list[DocumentChunk] = []
        for chunk_id, doc, meta, dist in zip(ids, documents, metadatas, distances):
            # Con hnsw:space='cosine', la distancia esta en [0, 2]; similitud = 1 - distancia
            similarity = max(0.0, round(1.0 - float(dist), 4))
            if similarity < min_score:
                continue
            matched.append(
                DocumentChunk(
                    id=chunk_id,
                    content=doc,
                    source=str(meta.get("source", "desconocido")),
                    page=int(meta.get("page", 0)),
                    file_hash=str(meta.get("file_hash", "")),
                    score=similarity,
                )
            )
        return matched

    def clear(self) -> None:
        self._client.delete_collection(name=self._collection_name)
        self._collection = self._get_or_create_collection()

    def count(self) -> int:
        return self._collection.count()

    def get_indexed_sources(self) -> dict[str, dict]:
        total = self.count()
        if total == 0:
            return {}

        data = self._collection.get(include=["metadatas"])
        sources: dict[str, dict] = {}
        for meta in data.get("metadatas", []) or []:
            if not meta:
                continue
            source = str(meta.get("source", "desconocido"))
            file_hash = str(meta.get("file_hash", ""))
            if source not in sources:
                sources[source] = {"file_hash": file_hash, "chunk_count": 0}
            sources[source]["chunk_count"] += 1
        return sources

    def delete_by_source(self, source: str) -> None:
        self._collection.delete(where={"source": source})
