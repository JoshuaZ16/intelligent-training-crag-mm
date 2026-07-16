"""Compatibility helpers for the public CRAG-MM search package."""

from __future__ import annotations

import json
import os
from typing import Any


class LazyCollectionMetadata:
    """Dictionary-like, cached metadata access backed by Chroma IDs."""

    def __init__(self, collection: Any):
        self.collection = collection
        self.cache: dict[str, dict] = {}

    def __getitem__(self, item_id: str) -> dict:
        key = str(item_id)
        if key not in self.cache:
            page = self.collection.get(ids=[key], include=["metadatas"])
            if not page["ids"]:
                raise KeyError(key)
            self.cache[key] = page["metadatas"][0]
        return self.cache[key]

    def __len__(self) -> int:
        return self.collection.count()


class LazyImageMetadata(LazyCollectionMetadata):
    """Load image metadata by Chroma ID and hydrate its entity attributes."""

    def __init__(self, collection: Any, entity_cache: dict[str, dict]):
        super().__init__(collection)
        self.entity_cache = entity_cache

    def __getitem__(self, item_id: int | str) -> dict:
        metadata = super().__getitem__(str(item_id))
        info = json.loads(metadata.get("info", "{}"))
        self.entity_cache.update(info)
        return metadata


def load_collection_metadata_paged(collection: Any, batch_size: int = 10_000) -> dict[str, dict]:
    """Load Chroma metadata without exceeding SQLite's variable limit."""
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")

    total = collection.count()
    metadata_by_id: dict[str, dict] = {}
    for offset in range(0, total, batch_size):
        limit = min(batch_size, total - offset)
        page = collection.get(limit=limit, offset=offset, include=["metadatas"])
        metadata_by_id.update(zip(page["ids"], page["metadatas"]))
    return metadata_by_id


def install_cragmm_lazy_web_metadata() -> None:
    """Patch cragmm-search-pipeline 0.5.1 to fetch only top-k metadata."""
    from cragmm_search.web_search_mock_api.api.web_index import CragMockWeb
    from cragmm_search.web_search_mock_api.api.web_search import index_web_data

    if getattr(CragMockWeb, "_course_paged_metadata", False):
        return

    def lazy_init(self, emb_model, tokenizer, text_index_path, web_hf_dataset_tag=None):
        self.vector_db = index_web_data(
            hf_path=text_index_path,
            revision=web_hf_dataset_tag,
        )
        self.emb_model = emb_model
        self.tokenizer = tokenizer
        self.index_to_metadata = LazyCollectionMetadata(self.vector_db)

    CragMockWeb.__init__ = lazy_init
    CragMockWeb._course_paged_metadata = True


def install_cragmm_lazy_image_metadata() -> None:
    """Patch image KG initialization to fetch metadata only for retrieved IDs."""
    import chromadb
    from huggingface_hub import snapshot_download

    from cragmm_search.image_search_mock_api.image_kg import CragImageKG

    if getattr(CragImageKG, "_course_lazy_metadata", False):
        return

    def lazy_init(
        self,
        emb_model,
        processor,
        hf_dataset_id,
        image_hf_dataset_tag=None,
    ):
        print(f"Loading image index from huggingface {hf_dataset_id}")
        dataset_local_path = snapshot_download(
            repo_id=hf_dataset_id,
            repo_type="dataset",
            revision=image_hf_dataset_tag,
        )
        client = chromadb.PersistentClient(path=dataset_local_path)
        self.vector_db = client.get_collection(name="image_embeddings")
        self.vector_db.modify(metadata={"hnsw:num_threads": os.cpu_count() or 1})
        self.kg = {}
        self.id2_data = LazyImageMetadata(self.vector_db, self.kg)
        self.emb_model = emb_model
        self.processor = processor

    CragImageKG.__init__ = lazy_init
    CragImageKG._course_lazy_metadata = True
