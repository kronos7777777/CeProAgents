# src/neo4j_sink.py
from neo4j import GraphDatabase

def ensure_constraints(cfg):
    if not cfg.get("create_constraints", False):
        return
    driver = GraphDatabase.driver(cfg["uri"], auth=(cfg["user"], cfg["password"]))
    with driver.session(database=cfg.get("database")) as s:
        s.run("""
        CREATE CONSTRAINT entity_id IF NOT EXISTS
        FOR (e:Entity) REQUIRE e.id IS UNIQUE;
        """)
    driver.close()

def neo4j_write_simple(entities, triplets, cfg, source="kg_pipeline"):
    """
    entities: List[str]
    triplets : List[{"subject":s,"relation":r,"object":o}]
    cfg     : dict from neo4j.yaml -> neo4j section
    """
    driver = GraphDatabase.driver(cfg["uri"], auth=(cfg["user"], cfg["password"]))
    with driver.session(database=cfg.get("database")) as sess:
        # 1) entity
        sess.run("""
        UNWIND $ids AS id
        MERGE (:Entity {id:id})
        """, ids=entities)

        # 2) triplets（constant relation :REL，relation details in type）
        rows = [{"s": t["subject"], "r": t["relation"], "o": t["object"]} for t in triplets]
        sess.run("""
        UNWIND $rows AS row
        MATCH (s:Entity {id: row.s})
        MATCH (o:Entity {id: row.o})
        MERGE (s)-[rel:REL {type: row.r}]->(o)
        ON CREATE SET rel.source = $source
        """, rows=rows, source=source)
    driver.close()
