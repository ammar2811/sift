"""Shared logic between the API and the ingestion worker.

Chunking, embedding, and retrieval live here rather than in either app so that
ingest-time and query-time can never drift apart - a chunking mismatch between the
two is the classic silent RAG bug.
"""
