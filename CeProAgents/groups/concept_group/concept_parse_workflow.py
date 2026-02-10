import autogen
import json
import base64
import io
import os
from PIL import Image
from .concept_utils import clean_and_parse_json
from .concept_prompts import (
    PARSR_EXPERT_PROMPT,
    COMPLETER_EXPERT_PROMPT,
    CORRECTOR_EXPERT_PROMPT,
    GENERATOR_EXPERT_PROMPT,
    CONCEPT_MANAGER_PROMPT,
    EQUIPMENT_EXTRACTOR_PROMPT,
    CONNECTION_EXTRACTOR_PROMPT,
    COMBINER_PROMPT

)

class ConceptParsingInternalWorkflow:
    """
    Internal Workflow Engine: Encapsulates the multi-step vision task 
    (Equipment Extraction -> Connection Extraction -> Graph Combination) as an atomic operation.
    """
    def __init__(self, llm_config):
        self.context = {"base64_image": "", "equipments": [], "connections": []}
        
        self.equipment_extractor = autogen.AssistantAgent(
            name="Inner_Equipment_Extractor", 
            llm_config=llm_config,
            system_message=EQUIPMENT_EXTRACTOR_PROMPT,
        )
        
        self.connection_extractor = autogen.AssistantAgent(
            name="Inner_Connection_Extractor", 
            llm_config=llm_config,
            system_message=CONNECTION_EXTRACTOR_PROMPT,
        )
        
        self.combiner = autogen.AssistantAgent(
            name="Inner_Graph_Combiner", 
            llm_config=llm_config,
            system_message=COMBINER_PROMPT,
        )
        
        self.internal_admin = autogen.UserProxyAgent(
            name="Inner_Admin", 
            human_input_mode="NEVER", 
            code_execution_config=False,
            max_consecutive_auto_reply=0
        )

    # def _encode(self, path):
    #     if not os.path.exists(path): return None
    #     Image.MAX_IMAGE_PIXELS = None 
    #     with Image.open(path) as img:
    #         byte_io = io.BytesIO()
    #         img.save(byte_io, format='PNG')
    #         return base64.b64encode(byte_io.getvalue()).decode('utf-8')
    def _encode(self, path):
        if not os.path.exists(path): 
            return None
    
        Image.MAX_IMAGE_PIXELS = None 
    
        with Image.open(path) as img:
            max_limit = 8000  
        
            width, height = img.size
            if width > max_limit or height > max_limit:
                scaling_factor = max_limit / max(width, height)
                new_size = (int(width * scaling_factor), int(height * scaling_factor))
            
                img = img.resize(new_size, Image.Resampling.LANCZOS)
                print(f"Image resized from {width}x{height} to {new_size}")

                if img.mode == 'RGBA':
                    img = img.convert('RGB')

            byte_io = io.BytesIO()
            img.save(byte_io, format='PNG') 
            return base64.b64encode(byte_io.getvalue()).decode('utf-8')

    def _msg_equipments(self, *args, **kwargs):
        return {
            "content": [
                {"type": "text", "text": "Extract Equipments from the diagram. Return JSON list only."}, 
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{self.context['base64_image']}"}}
            ]
        }

    def _msg_connections(self, *args, **kwargs):
        return {
            "content": [
                {"type": "text", "text": f"Context - Existing Equipments: {json.dumps(self.context['equipments'])}\nNow extract Connections based on the image. Return JSON list only."}, 
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{self.context['base64_image']}"}}
            ]
        }

    def _msg_combine(self, *args, **kwargs):
        return f"Combine these into one JSON object:\nEquipments: {json.dumps(self.context['equipments'])}\nConnections: {json.dumps(self.context['connections'])}"

    def run(self, image_path: str) -> str:
        print(f"\n>>> [Tool] Starting Internal Workflow for image: {image_path}")
        b64 = self._encode(image_path)
        if not b64: return json.dumps({"error": "Image file not found"})
        
        self.context['base64_image'] = b64

        try:
            # Step 1: Extract Equipments
            self.internal_admin.initiate_chat(self.equipment_extractor, message=self._msg_equipments, max_turns=1)
            res = self.internal_admin.last_message(self.equipment_extractor)["content"]
            self.context['equipments'] = clean_and_parse_json(res)

            # Step 2: Extract Connections
            self.internal_admin.initiate_chat(self.connection_extractor, message=self._msg_connections, max_turns=1)
            res = self.internal_admin.last_message(self.connection_extractor)["content"]
            print(f"Extracted Connections: {res}")
            self.context['connections'] = clean_and_parse_json(res)
            print(f"Parsed Connections: {self.context['connections']}")

            # Step 3: Combine
            self.internal_admin.initiate_chat(self.combiner, message=self._msg_combine, max_turns=1)
            final_res = self.internal_admin.last_message(self.combiner)["content"]
            
            final_obj = clean_and_parse_json(final_res)
            print(f"<<< [Tool] Parsing complete. Found {len(final_obj.get('equipments', []))} equipments.\n")
            return json.dumps(final_obj, ensure_ascii=False)
        except Exception as e:
            print(f"!!! Tool Error: {e}")
            return json.dumps({"error": str(e)})

# Singleton instance
_parsing_engine = ConceptParsingInternalWorkflow()

# === Public Tool Function ===
def execute_pid_parsing_tool(image_path: str, llm_config: dict) -> str:
    """
    Parses a PID image file using a multi-agent visual workflow and returns the graph structure (equipments/connections) in JSON.
    """
    return _parsing_engine.run(image_path, llm_config)