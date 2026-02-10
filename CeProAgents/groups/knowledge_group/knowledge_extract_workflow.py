import os
import json
import logging
import base64
import tqdm
import yaml
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Set
from time import perf_counter
from collections import defaultdict, Counter

import numpy as np
from autogen import AssistantAgent, UserProxyAgent, ConversableAgent

from configs import *

from .knowledge_prompts import (
    MERGE_SYSTEM_PROMPT,
    UNIFIED_SYSTEM_PROMPT 
)

from .knowledge_extraction import utils
from .knowledge_utils import save_extracted_knowledge
from .knowledge_extraction.neo4j_sink import (
    neo4j_write_simple,
    ensure_constraints,
)
from .knowledge_extraction.graph_io import export_pred_kg
from .knowledge_extraction.sem_embed import embed_texts, cosine_matrix
from .knowledge_extraction.union_find import UnionFind
from .knowledge_extraction.global_store import (
    open_global_assets, save_and_close,
    batch_search_texts, norm_text, record_llm_equivalences
)

# Initialize module logger
logger = logging.getLogger(__name__)


def clean_and_parse_json(json_str: str, context: str = "") -> Any:
    """Helper to parse JSON from LLM output."""
    try:
        return utils.parse_json_output(json_str, context=context)
    except Exception:
        try:
            json_str = json_str.strip()
            if json_str.startswith("```json"): json_str = json_str[7:]
            if json_str.endswith("```"): json_str = json_str[:-3]
            return json.loads(json_str)
        except json.JSONDecodeError as e:
            logger.error(f"JSON Parse Error in {context}: {e}")
            return [] if "list" in context.lower() else {}

class KnowledgeExtractionInternalWorkflow:
    """
    Internal Workflow Engine: Encapsulates the multi-step Knowledge Extraction task.
    Adopts the '_create_*_message' pattern using EXACT prompts from provided YAMLs.
    """

    def __init__(self, llm_config: Dict[str, Any]):
        
        self.llm_config = llm_config
        self.context: Dict[str, Any] = {
            "text_passage": "",
            "images_b64": [],
            "raw_entities": [],
            "raw_relations": []
        }

        self.unified_extractor = AssistantAgent(
            name="Unified_Triplet_Extractor",
            llm_config=llm_config,
            system_message=UNIFIED_SYSTEM_PROMPT 
        )

        self.merge_assistant = AssistantAgent(
            name="Inner_Merge_Assistant",
            llm_config=llm_config,
            system_message=MERGE_SYSTEM_PROMPT
        )

        self.internal_admin = UserProxyAgent(
            name="Inner_Admin",
            human_input_mode="NEVER",
            code_execution_config=False,
            max_consecutive_auto_reply=0
        )

        self.write_neo4j = WRITE_NEO4J
        self.neo4j_cfg = NEO4J_CONFIG
    
    def _create_text_triplet_extraction_message(self, chunk_text: str) -> Dict[str, Any]:
        """
        Creates a text-only payload for DIRECT triplet extraction.
        """
        prompt_text = f"""1.  **Task**: Your task is to act as a chemical engineering expert. Carefully analyze the provided text passage to extract knowledge triplets. A triplet consists of a subject, a relation, and an object `(subject, relation, object)`.

        2.  **Input**:
                text_passage: A text passage from chemical engineering literature.

        3.  **Output Format**: Provide the output as a SINGLE JSON list of objects. Each object represents one relationship triplet: `{{"subject": "...", "relation": "...", "object": "..."}}`.

        4.  **Predefined Relation List (CRITICAL: MUST USE ONLY THESE RELATIONS)**:
                - **Process Hierarchy & Origin**: `has_production_method_category`, `has_sub_category`, `has_variant`, `developed_by`, `has_alias`.
                - **Product Grade & Quality**: `produces_grade`, `has_purity_requirement`, `has_impurity_limit`.
                - **Process Flow**: `has_first_step`, `next_step`, `has_specific_process`, `includes`.
                - **Reaction & Chemistry**: `chemical_reaction_equation`, `has_feedstock`, `has_catalyst`, `produces`, `co_product`, `reaction_yield`, `conversion_rate`, `selectivity`.
                - **Conditions & Equipment**: `has_temperature`, `has_pressure`, `has_equipment`, `steam_consumption`, `reaction_phase`.
                - **Evaluation**: `has_advantage`, `has_disadvantage`, `technical_issue`.

        5.  **Instructions**:
                (1). **Identify Entities and Relations Simultaneously**: Read the text. When you identify a relationship, extract the subject, object, and the corresponding relation from the predefined list.
                (2). **Exact Wording**: The "subject" and "object" you extract MUST use the exact wording found in the text.
                (3). **Strict Relation Schema**: The "relation" field in your output MUST be one of the values from the 'Predefined Relation List' in section 4. Do not invent new relations.
                (4). **JSON Only**: Ensure the final output is ONLY the JSON list of relationship objects, without any introductory text, explanations, or markdown code blocks.

        6.  **Input Context**:
                text_passage: {chunk_text}"""
        
        return {"role": "user", "content": prompt_text}


    def _create_multimodal_triplet_extraction_message(self, chunk_text: str, relevant_images_b64: List[str]) -> Dict[str, Any]:
        """
        Creates a multimodal payload for DIRECT triplet extraction (Text + Image).
        """
        prompt_text = f"""1.  **Task**: Your task is to act as a chemical engineering expert. Carefully analyze the provided text passage and any accompanying images to extract knowledge triplets. A triplet consists of a subject, a relation, and an object `(subject, relation, object)`.

        2.  **Input**:
                text_passage: A text passage from chemical engineering literature.
                images: Attached technical diagrams (if any).

        3.  **Output Format**: Provide the output as a SINGLE JSON list of objects. Each object represents one relationship triplet: `{{"subject": "...", "relation": "...", "object": "..."}}`.

        4.  **Predefined Relation List (CRITICAL: MUST USE ONLY THESE RELATIONS)**:
                - **Process Hierarchy & Origin**: `has_production_method_category`, `has_sub_category`, `has_variant`, `developed_by`, `has_alias`.
                - **Product Grade & Quality**: `produces_grade`, `has_purity_requirement`, `has_impurity_limit`.
                - **Process Flow**: `has_first_step`, `next_step`, `has_specific_process`, `includes`.
                - **Reaction & Chemistry**: `chemical_reaction_equation`, `has_feedstock`, `has_catalyst`, `produces`, `co_product`, `reaction_yield`, `conversion_rate`, `selectivity`.
                - **Conditions & Equipment**: `has_temperature`, `has_pressure`, `has_equipment`, `steam_consumption`, `reaction_phase`.
                - **Evaluation**: `has_advantage`, `has_disadvantage`, `technical_issue`.

        5.  **Instructions**:
                (1). **Identify Entities and Relations Simultaneously**: Read the text and look at the images. When you identify a relationship, extract the subject, object, and the corresponding relation from the predefined list.
                (2). **Exact Wording**: The "subject" and "object" you extract MUST use the exact wording found in the text or image labels.
                (3). **Strict Relation Schema**: The "relation" field in your output MUST be one of the values from the 'Predefined Relation List' in section 4. Do not invent new relations.
                (4). **Comprehensive Extraction**: Extract all relevant triplets supported by the provided context (text and images). Arrows in diagrams often imply `next_step` or a process flow relationship.
                (5). **JSON Only**: Ensure the final output is ONLY the JSON list of relationship objects, without any introductory text, explanations, or markdown code blocks.

        6.  **Input Context**:
                text_passage: {chunk_text}"""

        content_payload = [{"type": "text", "text": prompt_text}]
        for img_b64 in relevant_images_b64:
            content_payload.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{img_b64}"}
            })

        return {"role": "user", "content": content_payload}

    def _create_merge_partition_message(self, names: List[str]) -> Dict[str, Any]:
        """Creates payload for entity partition"""
        numbered = "\n".join([f"{i+1}. {n}" for i, n in enumerate(names)])
        # ... [prompt content remains the same] ...
        return {
            "role": "user",
            "content": f"""1. Task:
     Your task is to review a small list of possibly similar entity names and partition them into equivalence subsets where items refer to the EXACT SAME real-world entity (true synonyms / aliases). Do NOT merge merely related, broader, narrower, or overlapping concepts.

  2. Input:
     Names (numbered): 
     {numbered}
     (Indices are 1-based and correspond exactly to the list above.)

  3. Output Format:
     Return a single RAW JSON array (no code fences, no explanations), where each element is an object:
     [
       {{"members":[<1-based indices>], "canonical":"<a string taken from the list>"}}
     ]

  4. Instructions:
     (1) Only group names that are true synonyms of the same entity (strict equivalence). 
         Do NOT merge terms that are merely related (e.g., water vs liquid), parent/child categories, processes vs materials, or properties vs entities.
     (2) "members" must be a non-empty list of 1-based indices, each index pointing to an item in the numbered list above. 
         If no synonyms exist, return [].
     (3) "canonical" must be a name that APPEARS IN THE LIST. Prefer a widely accepted, human-readable form when present:
         • Chemical formula ↔ common name: pick the common name if both appear (e.g., water over H2O), unless the formula is the most standard reference in this set.
         • Abbreviation ↔ full name: pick the full, expanded name when present (e.g., Polypropylene over PP; United States over USA).
         • Minor surface variants (case, hyphenation, whitespace) are synonyms.
     (4) You may use general knowledge and standard domain conventions to judge equivalence (e.g., H2O≡water; USA≡United States). 
         However, do not invent mappings beyond standard aliasing.
     (5) Do not output overlapping groups. Each index should appear in at most one "members" list.
     (6) Return only the JSON array. No extra keys, no comments, no code fences.

  5. Examples:
     Input (numbered):
     1. water
     2. H2O
     3. liquid
     Expected Output:
     [{{"members":[1,2], "canonical":"water"}}]

     Input (numbered):
     1. United States
     2. USA
     3. United States of America
     Expected Output:
     [{{"members":[1,2,3], "canonical":"United States"}}]"""
        }

    def _create_global_select_message(self, query: str, candidates: List[Tuple]) -> Dict[str, Any]:
        """Creates payload for global entity alignment"""
        numbered = "\n".join([f"{i+1}. {surf} ({cid})" for i, (cid, surf, _) in enumerate(candidates)])
        # ... [prompt content remains the same] ...
        return {
            "role": "user",
            "content": f"""1. Task:
     Given a QUERY entity name and a numbered list of CANDIDATE GLOBAL ENTITIES,
     pick exactly one candidate that is an exact alias/synonym of the query.
     If none matches, output 0.

  2. Input:
     QUERY: {query}
     CANDIDATES (numbered):
     {numbered}

  3. Output:
     Return STRICT JSON only:
     {{"choice": <index from the list or 0>}}"""
        }

    # ==========================================================================
    # Logic Implementation
    # ==========================================================================

    def _parse_pdf(self, pdf_path: str, output_dir: str) -> Tuple[str, List[str]]:
        """Parses PDF into text and images."""
        # ... [function body remains the same] ...
        print(">> Parsing PDF...")
        utils.parse_pdf_mineru(pdf_path, output_dir)
        pdf_name = os.path.splitext(os.path.basename(pdf_path))[0]
        
        text_md = os.path.join(output_dir, "text", f"{pdf_name}.md")
        text_passage = ""
        if os.path.exists(text_md):
            with open(text_md, "r", encoding="utf-8") as f:
                text_passage = f.read()
        
        images_dir = os.path.join(output_dir, "images")
        images_b64 = []
        if os.path.isdir(images_dir):
            names = sorted(os.listdir(images_dir), key=lambda x: (len(x), x))
            for name in names:
                p = os.path.join(images_dir, name)
                if os.path.isfile(p) and os.path.splitext(p)[1].lower() in SUPPORTED_IMAGE_EXTS:
                    images_b64.append(utils.encode_image(p))
        return text_passage, images_b64

    def _match_images_to_chunk(self, chunk_text: str, image_map: Dict[str, str]) -> List[str]:
        """Finds images relevant to a text chunk."""
        # ... [function body remains the same] ...
        relevant_b64 = []
        for filename, b64_data in image_map.items():
            base_name = os.path.splitext(filename)[0]
            if base_name in chunk_text or filename in chunk_text:
                relevant_b64.append(b64_data)
        return relevant_b64
    
    def _extract_text_knowledge_triplets(self, text_passage: str):
        """
        Single-pass knowledge extraction for text-only models.
        """
        chunks = utils.chunk_text(text_passage, device='cuda')
        logger.info(f"Processing {len(chunks)} chunks using Text-Only Direct Triplet Extractor...")

        for chunk in tqdm.tqdm(chunks, total=len(chunks), desc="Text Triplet Extraction"):
            msg = self._create_text_triplet_extraction_message(chunk)
            self.internal_admin.initiate_chat(self.unified_extractor, message=msg, max_turns=1, silent=SILENT)
            
            res_content = self.internal_admin.last_message(self.unified_extractor)["content"]
            chunk_triplets = clean_and_parse_json(res_content, "direct text triplets")
            
            if not isinstance(chunk_triplets, list):
                continue
            
            chunk_entities = set()
            valid_triplets = []
            
            for triplet in chunk_triplets:
                if isinstance(triplet, dict) and 'subject' in triplet and 'object' in triplet:
                    subject = str(triplet['subject']).strip()
                    object_ = str(triplet['object']).strip()
                    
                    if subject and object_:
                        chunk_entities.add(subject)
                        chunk_entities.add(object_)
                        valid_triplets.append(triplet)

            self.context['raw_entities'].extend(list(chunk_entities))
            self.context['raw_relations'].extend(valid_triplets)

    def _extract_multimodal_knowledge(self, text_passage: str, images_b64: List[str], output_dir: str):
        """
        Single-pass knowledge extraction for multimodal models.
        """
        images_dir = os.path.join(output_dir, "images")
        image_map = {}
        if os.path.exists(images_dir) and images_b64:
            names = sorted([n for n in os.listdir(images_dir) if os.path.isfile(os.path.join(images_dir, n)) and os.path.splitext(n)[1].lower() in SUPPORTED_IMAGE_EXTS], key=lambda x: (len(x), x))
            limit = min(len(names), len(images_b64))
            for i in range(limit):
                image_map[names[i]] = images_b64[i]

        chunks = utils.chunk_text(text_passage, device='cuda')
        logger.info(f"Processing {len(chunks)} chunks using Multimodal Direct Triplet Extractor...")

        for i, chunk in tqdm.tqdm(enumerate(chunks), total=len(chunks), desc="Multimodal Triplet Extraction"):

            try:
                relevant_imgs = self._match_images_to_chunk(chunk, image_map)
                
                msg = self._create_multimodal_triplet_extraction_message(chunk, relevant_imgs)
                self.internal_admin.initiate_chat(self.unified_extractor, message=msg, max_turns=1, silent=SILENT)
                
                res_content = self.internal_admin.last_message(self.unified_extractor)["content"]
                chunk_triplets = clean_and_parse_json(res_content, "direct multimodal triplets")
            
            except:
                continue

            if not isinstance(chunk_triplets, list):
                continue

            chunk_entities = set()
            valid_triplets = []
            
            for triplet in chunk_triplets:
                if isinstance(triplet, dict) and 'subject' in triplet and 'object' in triplet:
                    subject = str(triplet['subject']).strip()
                    object_ = str(triplet['object']).strip()
                    if subject and object_:
                        chunk_entities.add(subject)
                        chunk_entities.add(object_)
                        valid_triplets.append(triplet)

            self.context['raw_entities'].extend(list(chunk_entities))
            self.context['raw_relations'].extend(valid_triplets)
            
    def _lexical_ok(self, a, b):
        try: return norm_text(a) == norm_text(b)
        except: return a.strip().lower() == b.strip().lower()

    def _dedupe_local(self, entities: List[str]):
        
        entities = [e for e in entities if e.strip()]
        uniq, seen = [], set()
        for s in entities:
            if s not in seen: seen.add(s); uniq.append(s)
        n = len(uniq)
        if n <= 1: return uniq, {e: e for e in entities}
        print(f">> Computing embeddings for {n} unique entities...")
        X = embed_texts(uniq).astype("float32")
        import faiss
        index = faiss.IndexFlatIP(X.shape[1])
        index.add(X)
        D, I = index.search(X, min(MERGE_TOPK, n)) 

        uf = UnionFind(n)
        decision_cache = {}
        suggestions = defaultdict(list)
        
        call = 0
        for i in tqdm.tqdm(range(n), desc="Local Deduplication"):
            # print("\n") # Commented out to reduce noise with progress bar
            # print(f"{n}: Local Deduplication LLM Calls: {call}", end="\n")  
            sims = D[i].tolist()
            idxs = I[i].tolist()
            if i in idxs: k = idxs.index(i); idxs.pop(k); sims.pop(k)
            
            if idxs and sims[0] >= MERGE_AUTO_TAU and self._lexical_ok(uniq[i], uniq[idxs[0]]):
                uf.union(i, idxs[0])
                continue
            
            cand = [(j, s) for j, s in zip(idxs, sims) if s >= MERGE_REVIEW_TAU]
            undecided = [(j, s) for j, s in cand if (min(i, j), max(i, j)) not in decision_cache]
            if not undecided: 
                continue

            group_names = [uniq[i]] + [uniq[j] for j, _ in undecided]
            
            msg = self._create_merge_partition_message(group_names)
            call += 1
            self.internal_admin.initiate_chat(self.merge_assistant, message=msg, max_turns=1, silent=SILENT)
            res = self.internal_admin.last_message(self.merge_assistant)["content"]

            partitions = clean_and_parse_json(res, "merge partition")
            
            if isinstance(partitions, list):
                same_set = set()
                cano = None
                for p in partitions:
                    mem = [int(m) for m in p.get("members", []) if isinstance(m, (int, float))]
                    if 1 in mem:
                        same_set = {m-1 for m in mem}
                        cano = p.get("canonical")
                        break
                if cano: suggestions[i].append(cano)
                for k, (j, s) in enumerate(undecided, start=1):
                    pair = (min(i, j), max(i, j))
                    is_same = (k in same_set)
                    decision_cache[pair] = is_same
                    if is_same:
                        uf.union(i, j)
                        if cano: suggestions[j].append(cano)

        comps = uf.components()
        entity_map_unique = {}
        for comp in comps:
            pool = []
            for idx in comp: pool.extend(suggestions.get(idx, []))
            canonical = Counter(pool).most_common(1)[0][0] if pool else uniq[min(comp)]
            for idx in comp: entity_map_unique[uniq[idx]] = canonical
            
        local_map = {e: entity_map_unique.get(e, e) for e in entities}
        local_list = sorted(list(set(str(v) for v in local_map.values())))

        return local_list, local_map

    def _align_to_global(self, local_list, local_map):
        print(">> Opening global assets for alignment...")
        index, db, meta, paths = open_global_assets()
        premap = {}
        pending = []

        for name in local_list:
            hit = db.get_alias(norm_text(name))
            if hit: premap[name] = hit[0]
            else: pending.append(name)
            
        if pending and getattr(index, "ntotal", 0) > 0:
            D, I = batch_search_texts(index, pending, topk=GLOBAL_ALIGN_TOPK, batch=GLOBAL_ALIGN_BATCH)
            all_ids = sorted({int(x) for x in I.flatten() if int(x) >= 0})
            vec2meta = {}
            if all_ids:
                 q = f"SELECT vec_id, canonical_id, surface FROM vectors WHERE alive=1 AND vec_id IN ({','.join(['?']*len(all_ids))})"
                 rows = db.conn.execute(q, all_ids).fetchall()
                 vec2meta = {int(vid): (cid, surf) for (vid, cid, surf) in rows}

            for r, qname in tqdm.tqdm(enumerate(pending), total=len(pending), desc="Global Alignment"):
                sims = D[r]
                vids = I[r]
                candidates = []
                for vid, s in zip(vids, sims):
                    if int(vid) in vec2meta and s >= GLOBAL_ALIGN_THRESHOLD:
                        cid, surf = vec2meta[int(vid)]
                        candidates.append((cid, surf, s))
                
                if candidates:
                    candidates.sort(key=lambda x: x[2], reverse=True)
                    msg = self._create_global_select_message(qname, candidates)
                    self.internal_admin.initiate_chat(self.merge_assistant, message=msg, max_turns=1, silent=SILENT)
                    res = self.internal_admin.last_message(self.merge_assistant)["content"]
                    parsed = clean_and_parse_json(res, "global select")
                    if isinstance(parsed, dict):
                        choice = int(parsed.get("choice", 0))
                        if 1 <= choice <= len(candidates):
                            premap[qname] = candidates[choice-1][1]
        
        final_map = {}
        for orig, local_c in local_map.items():
            final_map[orig] = premap.get(local_c, local_c)
        save_and_close(index, db, paths)

        sorted_values = sorted(list(set(str(v) for v in final_map.values())))
        return sorted_values, final_map 

    def _persist_to_global(self, entities, entity_map):
        if not GLOBAL_STORE_WRITE: return
        print(">> Persisting new entities to global store...")
        index, db, meta, paths = open_global_assets()
        for c in entities:
             record_llm_equivalences(index, db, c, [c], also_add_vectors=True, write_alias=True)
        
        aliases = defaultdict(list)
        for orig, cano in entity_map.items():
            if orig != cano: aliases[cano].append(orig)
        for cano, names in aliases.items():
            to_add = [n for n in names if not db.has_vector_for_surface(n)]
            if to_add:
                record_llm_equivalences(index, db, cano, to_add, also_add_vectors=True, write_alias=True)
                
        save_and_close(index, db, paths)

    def _standardize_triplets(self, raw_rels, entity_map):
        standardized = []
        seen = set()
        for t in raw_rels:
            s = entity_map.get(t.get('subject'), t.get('subject'))
            o = entity_map.get(t.get('object'), t.get('object'))
            r = t.get('relation')
            s = str(s).strip()
            o = str(o).strip()
            r = str(r).strip()
            if s and o and s.lower() != o.lower():
                key = (s, r, o)
                if key not in seen:
                    seen.add(key)
                    standardized.append({'subject': s, 'relation': r, 'object': o})
        return standardized
        
    def run(self, input_path: str, output_dir: str) -> str:
        logger.info(f"Starting Knowledge Workflow: {input_path}")
        print(f"\n{'='*40}\n[Step 1/5] Starting Workflow for: {os.path.basename(input_path)}\n{'='*40}")
        if not input_path.lower().endswith(".pdf"): return json.dumps({"error": "PDF only"})
        
        text_passage, images_b64 = self._parse_pdf(input_path, output_dir)
        print(f">> PDF Parsed. Text length: {len(text_passage)} chars, Images: {len(images_b64)}")
        
        # MODIFIED: Reverted to your original logic for model dispatching.
        # Make sure UNIFIED_EXTRACTOR_MODELS is defined in your `configs.py`.
        print(f"\n{'='*40}\n[Step 2/5] Extracting Knowledge...\n{'='*40}")
        
        model_name = self.llm_config['config_list'][0].get('model', '')
        
        # This logic now exactly mirrors your original code's intention
        if model_name in UNIFIED_EXTRACTOR_MODELS:
            # This branch handles models you've designated as capable of multimodal input
            if images_b64:
                print(f">> Model '{model_name}' is a unified model. Processing text and images.")
                self._extract_multimodal_knowledge(text_passage, images_b64, output_dir)
            else:
                # If it's a multimodal model but no images exist, fall back to text-only processing
                print(f">> Model '{model_name}' is a unified model, but no images found. Processing text only.")
                self._extract_text_knowledge_triplets(text_passage)
        else:
            # This branch handles all other models (e.g., deepseek) as text-only
            print(f">> Model '{model_name}' is a text-only model. Processing text only.")
            self._extract_text_knowledge_triplets(text_passage)

        all_ents = self.context['raw_entities']
        all_rels = self.context['raw_relations']
        print(f">> Extraction done. Raw Entities: {len(all_ents)}, Raw Relations: {len(all_rels)}")
        
        print(f"\n{'='*40}\n[Step 3/5] Merging & Deduplicating Local Entities...\n{'='*40}")
        local_list, local_map = self._dedupe_local(all_ents)
        print(f">> Deduplication done. Unique Local Entities: {len(local_list)}")
        
        if GLOBAL_STORE_MATCH:
            print(f"\n{'='*40}\n[Step 4/5] Aligning to Global Knowledge Store...\n{'='*40}")
            final_ents, final_map = self._align_to_global(local_list, local_map)
        else:
            final_ents, final_map = local_list, local_map
            
        final_triplets = self._standardize_triplets(all_rels, final_map)
        self._persist_to_global(final_ents, final_map)
        
        print(f"\n{'='*40}\n[Step 5/5] Saving Results...\n{'='*40}")
        export_pred_kg(os.path.basename(input_path), final_ents, final_triplets, output_dir)
        if self.write_neo4j and self.neo4j_cfg:
            print(">> Writing to Neo4j...")
            neo4j_write_simple(final_ents, final_triplets, self.neo4j_cfg)
            
        triplets_list = [[t['subject'], t['relation'], t['object']] for t in final_triplets]
        payload = {
            "raw_text": text_passage,
            "triplets": triplets_list,
            "stats": {"entities": len(final_ents), "relations": len(final_triplets)}
        }
        save_msg = save_extracted_knowledge(json.dumps(payload, ensure_ascii=False))
        
        print(f"\n>> All Done! {save_msg}")
        return f"Workflow Done. {save_msg}"