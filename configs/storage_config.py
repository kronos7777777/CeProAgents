import os

# =========================
# Storage Settings
# =========================
STORAGE_ROOT = "./data_storage"
CHROMA_DB_PATH = os.path.join(STORAGE_ROOT, "chroma_db")
GRAPH_FILE_PATH = os.path.join(STORAGE_ROOT, "knowledge_graph.graphml")
COLLECTION_NAME = "knowledge_rag_store"