"""
Unit tests for helper_functions/vector_store.py
Tests for SimpleVectorStore, Document, chunk_text, read_documents_from_directory
"""
import json
import os
import pytest
from unittest.mock import patch, Mock
from helper_functions.vector_store import (
    SimpleVectorStore, Document, RetrievalResult,
    cosine_similarity, chunk_text, read_documents_from_directory
)


class TestDocument:
    def test_creation_defaults(self):
        doc = Document(text="hello")
        assert doc.text == "hello"
        assert doc.metadata == {}
        assert doc.doc_id is not None

    def test_creation_with_metadata(self):
        doc = Document(text="test", metadata={"key": "val"}, doc_id="abc")
        assert doc.text == "test"
        assert doc.metadata == {"key": "val"}
        assert doc.doc_id == "abc"

    def test_get_content(self):
        doc = Document(text="content here")
        assert doc.get_content() == "content here"


class TestRetrievalResult:
    def test_creation(self):
        r = RetrievalResult(text="result", metadata={"k": "v"}, score=0.9, doc_id="id1")
        assert r.text == "result"
        assert r.score == 0.9
        assert r.doc_id == "id1"
        assert r.get_content() == "result"


class TestCosineSimilarity:
    def test_identical_vectors(self):
        v = [1.0, 2.0, 3.0]
        assert cosine_similarity(v, v) == pytest.approx(1.0)

    def test_orthogonal_vectors(self):
        assert cosine_similarity([1, 0], [0, 1]) == pytest.approx(0.0)

    def test_opposite_vectors(self):
        assert cosine_similarity([1, 0], [-1, 0]) == pytest.approx(-1.0)

    def test_zero_vector(self):
        assert cosine_similarity([0, 0], [1, 2]) == 0.0


class TestChunkText:
    def test_short_text_no_split(self):
        text = "This is short."
        chunks = chunk_text(text, chunk_size=1000)
        assert len(chunks) == 1
        assert chunks[0] == text

    def test_long_text_splits(self):
        text = "First sentence. Second sentence. Third sentence. Fourth sentence. Fifth sentence."
        chunks = chunk_text(text, chunk_size=40, overlap=10)
        assert len(chunks) > 1

    def test_empty_text(self):
        chunks = chunk_text("", chunk_size=100)
        assert chunks == []

    def test_single_sentence(self):
        text = "Just one sentence."
        chunks = chunk_text(text, chunk_size=10, overlap=5)
        assert len(chunks) >= 1


class TestReadDocumentsFromDirectory:
    def test_reads_txt_files(self, tmp_path):
        (tmp_path / "doc.txt").write_text("Hello world")
        docs = read_documents_from_directory(str(tmp_path))
        assert len(docs) == 1
        assert docs[0].text == "Hello world"
        assert docs[0].metadata["filename"] == "doc.txt"

    def test_reads_multiple_files(self, tmp_path):
        (tmp_path / "a.txt").write_text("First")
        (tmp_path / "b.txt").write_text("Second")
        docs = read_documents_from_directory(str(tmp_path))
        assert len(docs) == 2

    def test_skips_empty_files(self, tmp_path):
        (tmp_path / "empty.txt").write_text("")
        (tmp_path / "real.txt").write_text("Content")
        docs = read_documents_from_directory(str(tmp_path))
        assert len(docs) == 1

    def test_nonexistent_directory(self):
        docs = read_documents_from_directory("/nonexistent/path")
        assert docs == []


class TestSimpleVectorStore:
    @pytest.fixture
    def embed_fn(self):
        """Mock embedding function that returns a simple vector based on text length."""
        def fn(text, input_type="passage"):
            # Return a simple deterministic vector
            n = len(text) % 10
            return [float(n)] * 4
        return fn

    def test_create_empty_store(self, tmp_path):
        store = SimpleVectorStore(persist_dir=str(tmp_path / "store"))
        assert len(store) == 0

    def test_insert_and_persist(self, tmp_path, embed_fn):
        store_dir = str(tmp_path / "store")
        store = SimpleVectorStore(persist_dir=store_dir, embed_fn=embed_fn)
        doc = Document(text="test document", metadata={"key": "val"}, doc_id="d1")
        store.insert(doc)
        assert len(store) == 1

        # Verify persistence
        store2 = SimpleVectorStore(persist_dir=store_dir, embed_fn=embed_fn)
        assert len(store2) == 1

    def test_insert_many(self, tmp_path, embed_fn):
        store = SimpleVectorStore(persist_dir=str(tmp_path / "store"), embed_fn=embed_fn)
        docs = [
            Document(text="doc one", doc_id="d1"),
            Document(text="doc two", doc_id="d2"),
        ]
        store.insert_many(docs)
        assert len(store) == 2

    def test_search(self, tmp_path, embed_fn):
        store = SimpleVectorStore(persist_dir=str(tmp_path / "store"), embed_fn=embed_fn)
        store.insert(Document(text="machine learning", doc_id="d1"))
        store.insert(Document(text="deep learning", doc_id="d2"))
        results = store.search("learning", top_k=2)
        assert len(results) <= 2
        for r in results:
            assert isinstance(r, RetrievalResult)
            assert r.score is not None

    def test_search_results_ordered_by_similarity(self, tmp_path):
        """Test that search results are returned in descending similarity order."""
        # Use an embedding function that produces distinct vectors per text
        def directional_embed(text, input_type="passage"):
            if "artificial intelligence" in text.lower():
                return [0.9, 0.1, 0.0, 0.0]
            elif "machine learning" in text.lower():
                return [0.7, 0.3, 0.0, 0.0]
            elif "cooking recipes" in text.lower():
                return [0.0, 0.0, 0.9, 0.1]
            return [0.5, 0.5, 0.0, 0.0]  # default / query

        store = SimpleVectorStore(persist_dir=str(tmp_path / "store"), embed_fn=directional_embed)
        store.insert(Document(text="cooking recipes for dinner", doc_id="d1"))
        store.insert(Document(text="artificial intelligence research", doc_id="d2"))
        store.insert(Document(text="machine learning algorithms", doc_id="d3"))

        results = store.search("artificial intelligence overview", top_k=3)
        assert len(results) >= 2
        # Results must be in descending similarity order
        for i in range(len(results) - 1):
            assert results[i].score >= results[i + 1].score

    def test_search_empty_store(self, tmp_path, embed_fn):
        store = SimpleVectorStore(persist_dir=str(tmp_path / "store"), embed_fn=embed_fn)
        results = store.search("query", top_k=5)
        assert results == []

    def test_search_no_embed_fn(self, tmp_path):
        store = SimpleVectorStore(persist_dir=str(tmp_path / "store"))
        results = store.search("query")
        assert results == []

    def test_get_document(self, tmp_path, embed_fn):
        store = SimpleVectorStore(persist_dir=str(tmp_path / "store"), embed_fn=embed_fn)
        store.insert(Document(text="hello", metadata={"k": "v"}, doc_id="d1"))
        doc = store.get_document("d1")
        assert doc is not None
        assert doc.text == "hello"
        assert doc.doc_id == "d1"

    def test_get_document_not_found(self, tmp_path):
        store = SimpleVectorStore(persist_dir=str(tmp_path / "store"))
        assert store.get_document("nonexistent") is None

    def test_delete_document(self, tmp_path, embed_fn):
        store = SimpleVectorStore(persist_dir=str(tmp_path / "store"), embed_fn=embed_fn)
        store.insert(Document(text="to delete", doc_id="d1"))
        assert len(store) == 1
        assert store.delete_document("d1") is True
        assert len(store) == 0

    def test_delete_document_not_found(self, tmp_path):
        store = SimpleVectorStore(persist_dir=str(tmp_path / "store"))
        assert store.delete_document("nonexistent") is False

    def test_docs_property(self, tmp_path, embed_fn):
        store = SimpleVectorStore(persist_dir=str(tmp_path / "store"), embed_fn=embed_fn)
        store.insert(Document(text="doc1", doc_id="d1"))
        store.insert(Document(text="doc2", doc_id="d2"))
        docs = store.docs
        assert "d1" in docs
        assert "d2" in docs

    def test_from_documents(self, tmp_path, embed_fn):
        docs = [
            Document(text="first", doc_id="d1"),
            Document(text="second", doc_id="d2"),
        ]
        store = SimpleVectorStore.from_documents(docs, persist_dir=str(tmp_path / "store"), embed_fn=embed_fn)
        assert len(store) == 2

    def test_corrupted_store_file(self, tmp_path):
        store_dir = tmp_path / "store"
        store_dir.mkdir()
        (store_dir / "vector_store.json").write_text("not valid json")
        store = SimpleVectorStore(persist_dir=str(store_dir))
        assert len(store) == 0
