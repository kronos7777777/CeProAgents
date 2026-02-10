from CeProAgents.core.database import db
from ddgs import DDGS
import json

def save_extracted_knowledge(json_str: str):
    try:
        data = json.loads(json_str)
        raw_text = data.get("raw_text", "")
        if raw_text:
            db.add_document(raw_text)
        
        triplets = data.get("triplets", []) # [[head, rel, tail], ...]
        count = 0
        for h, r, t in triplets:
            db.add_kg_triplet(h, r, t)
            count += 1
        return f"TERMINATE. Saved {count} relations and document content."
    except Exception as e:
        return f"Error saving knowledge: {str(e)}"

def search_kg(entity_name: str):
    results = db.get_kg_subgraph(entity_name)
    if not results:
        return "No direct relations found in KG."
    return "\n".join(results)

def search_rag(query: str):
    results = db.query_vector(query)
    if not results:
        return "No relevant documents found in vector store."
    return "\n".join(results)

def search_web(query: str):
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, region='wt-wt', max_results=3))
            return json.dumps(results, ensure_ascii=False)
    except Exception as e:
        return f"Web search failed: {str(e)}"