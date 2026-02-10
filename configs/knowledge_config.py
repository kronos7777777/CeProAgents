# knowledge_config.py

"""
Centralized configuration for Knowledge Extraction & Retrieval Workflow.
Hardcoded settings.
"""

# Supported image formats for extraction
SUPPORTED_IMAGE_EXTS = {".jpg", ".png", ".jpeg"}

# ==============================================================================
# 2. Neo4j Settings
# ==============================================================================
# Whether to write results to Neo4j (True / False)
WRITE_NEO4J = False

# Path to Neo4j config yaml. 
# Leave empty "" to rely on internal default paths or disable if not found.
NEO4J_CONFIG = {
    
    "uri": " ",
    "user": "neo4j",
    "password": " ",
    "database": "neo4j",
    "create_constraints": True
}

# ==============================================================================
# 3. Global Store Settings (FAISS + SQLite)
# ==============================================================================
# Whether to write new entities/aliases into Global Store
# Generally set to False during testing to avoid polluting the global DB.
GLOBAL_STORE_WRITE = True

# Whether to attempt matching extracted entities against Global Store
# Set to True to enable linking local entities to global canonical IDs.
GLOBAL_STORE_MATCH = True

# Parameters for Global Alignment search
GLOBAL_ALIGN_TOPK = 20
GLOBAL_ALIGN_BATCH = 512
GLOBAL_ALIGN_THRESHOLD = 0.95

# ==============================================================================
# 4. Merge & Deduplication Logic (Local)
# ==============================================================================
# Threshold for sending a candidate pair to LLM for review (0.0 - 1.0)
# Lower = more aggressive recall (more LLM calls), Higher = stricter
MERGE_REVIEW_TAU = 0.98

# Threshold for automatic merging WITHOUT LLM (0.0 - 1.0)
# Must be very high (e.g., 0.90+) to avoid errors. Also checks lexical similarity.
MERGE_AUTO_TAU = 0.99

# Number of neighbors to retrieve from FAISS for local deduplication
MERGE_TOPK = 5

GROUP_CHAT_MAX_ROUND = 20