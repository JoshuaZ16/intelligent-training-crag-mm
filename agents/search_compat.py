"""Compatibility helpers for the public CRAG-MM search package."""

from __future__ import annotations

import json
import os
from pathlib import Path
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


def resolve_dataset_snapshot(
    dataset_id_or_path: str,
    revision: str | None = None,
) -> str:
    """Resolve either an audited local snapshot or a pinned HF dataset."""
    local_path = Path(dataset_id_or_path).expanduser()
    if local_path.is_dir():
        return str(local_path.resolve())

    from huggingface_hub import snapshot_download

    return snapshot_download(
        repo_id=dataset_id_or_path,
        repo_type="dataset",
        revision=revision,
    )


def close_search_pipeline(pipeline: Any) -> list[str]:
    """Stop owned Chroma systems and release retrieval model references."""
    if pipeline is None or getattr(pipeline, "_course_closed", False):
        return []

    errors: list[str] = []
    seen_systems: set[int] = set()
    for owner_name in ("crag_image_kg", "text_web"):
        owner = getattr(pipeline, owner_name, None)
        client = getattr(owner, "_chroma_client", None)
        system = getattr(client, "_system", None)
        if system is not None and id(system) not in seen_systems:
            seen_systems.add(id(system))
            try:
                system.stop()
            except Exception as exc:  # pragma: no cover - defensive cleanup.
                errors.append(
                    f"{owner_name} Chroma stop failed: "
                    f"{type(exc).__name__}: {exc}"
                )
        if owner is not None:
            owner._chroma_client = None

    for attribute in (
        "image_collection",
        "crag_image_kg",
        "image_model",
        "image_processor",
        "text_model",
        "text_tokenizer",
        "text_web",
    ):
        if hasattr(pipeline, attribute):
            setattr(pipeline, attribute, None)
    pipeline._course_closed = True
    return errors


def install_cragmm_lazy_web_metadata() -> None:
    """Patch cragmm-search-pipeline 0.5.1 to fetch only top-k metadata."""
    import chromadb

    from cragmm_search.web_search_mock_api.api.web_index import CragMockWeb

    if getattr(CragMockWeb, "_course_paged_metadata", False):
        return

    def lazy_init(self, emb_model, tokenizer, text_index_path, web_hf_dataset_tag=None):
        dataset_local_path = resolve_dataset_snapshot(
            text_index_path,
            web_hf_dataset_tag,
        )
        self._chroma_client = chromadb.PersistentClient(
            path=dataset_local_path
        )
        self.vector_db = self._chroma_client.get_collection(
            name="web_search_embeddings"
        )
        self.vector_db.modify(
            metadata={"hnsw:num_threads": os.cpu_count() or 1}
        )
        self.emb_model = emb_model
        self.tokenizer = tokenizer
        self.index_to_metadata = LazyCollectionMetadata(self.vector_db)

    CragMockWeb.__init__ = lazy_init
    CragMockWeb._course_paged_metadata = True


def install_cragmm_lazy_image_metadata() -> None:
    """Patch image KG initialization to fetch metadata only for retrieved IDs."""
    import chromadb

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
        dataset_local_path = resolve_dataset_snapshot(
            hf_dataset_id,
            image_hf_dataset_tag,
        )
        self._chroma_client = chromadb.PersistentClient(
            path=dataset_local_path
        )
        self.vector_db = self._chroma_client.get_collection(
            name="image_embeddings"
        )
        self.vector_db.modify(metadata={"hnsw:num_threads": os.cpu_count() or 1})
        self.kg = {}
        self.id2_data = LazyImageMetadata(self.vector_db, self.kg)
        self.emb_model = emb_model
        self.processor = processor

    CragImageKG.__init__ = lazy_init
    CragImageKG._course_lazy_metadata = True
