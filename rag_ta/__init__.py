"""RAG-based AI Teaching Assistant.

Pipeline: video -> FFmpeg (audio) -> Whisper (timestamped transcript) -> chunks
-> multilingual embeddings -> FAISS -> retrieve top-k -> LLM answers with timestamps.
"""
