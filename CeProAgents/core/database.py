import os
import uuid
import logging
from typing import List, Dict, Any

import networkx as nx
import chromadb
from chromadb.api.models.Collection import Collection

from groups.knowledge_group.knowledge_extraction import utils

from .embedding import get_embedding

from configs import STORAGE_ROOT, CHROMA_DB_PATH, GRAPH_FILE_PATH, COLLECTION_NAME

logger = logging.getLogger(__name__)

class GlobalDatabase:
    _instance = None
    
    chroma_client: chromadb.PersistentClient
    vector_collection: Collection
    graph: nx.DiGraph

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(GlobalDatabase, cls).__new__(cls)
            cls._instance._initialize()
        return cls._instance

    def _initialize(self) -> None:
        if not os.path.exists(STORAGE_ROOT):
            os.makedirs(STORAGE_ROOT, exist_ok=True)

        logger.info("Initializing GlobalDatabase resources...")

        self.chroma_client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
        self.vector_collection = self.chroma_client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"}
        )

        self._load_graph()

    def _load_graph(self) -> None:
        if os.path.exists(GRAPH_FILE_PATH):
            try:
                self.graph = nx.read_graphml(GRAPH_FILE_PATH)
                logger.info(
                    f"Knowledge Graph loaded: {self.graph.number_of_nodes()} nodes, "
                    f"{self.graph.number_of_edges()} edges"
                )
            except Exception as e:
                logger.error(f"Graph file corrupted, creating new graph. Error: {e}")
                self.graph = nx.DiGraph()
        else:
            self.graph = nx.DiGraph()

    def add_rag_document(self, text: str, source: str = "unknown") -> None:
        if not text or not text.strip():
            logger.warning("Attempted to add empty document to RAG.")
            return

        try:
            text_chunks = utils.chunk_text(text, device='cuda')
            for text in text_chunks:
                vector = get_embedding(text)
                doc_id = str(uuid.uuid4())

                self.vector_collection.add(
                    documents=[text],
                    embeddings=[vector],
                    metadatas=[{"source": source}],
                    ids=[doc_id]
                )
        except Exception as e:
            logger.error(f"Failed to add RAG document: {e}")

    def search_rag(self, query: str, top_k: int = 3) -> List[str]:
        try:
            query_vec = get_embedding(query)
            
            results = self.vector_collection.query(
                query_embeddings=[query_vec],
                n_results=top_k
            )
            if results and results.get('documents'):
                return results['documents'][0]
            
            return []
        except Exception as e:
            logger.error(f"RAG search failed: {e}")
            return []

    def add_kg_triplet(self, head: str, relation: str, tail: str) -> None:
        self.graph.add_edge(head, tail, relation=relation)
        
        try:
            nx.write_graphml(self.graph, GRAPH_FILE_PATH)
        except Exception as e:
            logger.error(f"Failed to save Knowledge Graph to disk: {e}")

    def search_kg(self, entity: str) -> List[str]:
        if not self.graph.has_node(entity):
            return []

        relationships = []
        
        for neighbor in self.graph.successors(entity):
            edge_data = self.graph.get_edge_data(entity, neighbor)
            rel = edge_data.get('relation', 'related_to')
            relationships.append(f"{entity} --[{rel}]--> {neighbor}")
        
        for neighbor in self.graph.predecessors(entity):
            edge_data = self.graph.get_edge_data(neighbor, entity)
            rel = edge_data.get('relation', 'related_to')
            relationships.append(f"{neighbor} --[{rel}]--> {entity}")

        return relationships

    def get_kg_stats(self) -> Dict[str, int]:
        """Returns current statistics of the Knowledge Graph."""
        return {
            "nodes": self.graph.number_of_nodes(),
            "edges": self.graph.number_of_edges()
        }

    def add_document(self, text: str, source: str = "unknown") -> None:
        """Alias for add_rag_document."""
        self.add_rag_document(text, source=source)

    def query_vector(self, query: str, top_k: int = 3) -> List[str]:
        """Alias for search_rag."""
        return self.search_rag(query, top_k=top_k)

    def get_kg_subgraph(self, entity: str) -> List[str]:
        """Alias for search_kg."""
        return self.search_kg(entity)

db = GlobalDatabase()