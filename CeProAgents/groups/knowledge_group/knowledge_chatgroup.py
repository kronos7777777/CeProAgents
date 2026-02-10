# knowledge_chatgroup.py

import logging
from typing import Dict, Any, List, Optional, Union

import autogen
from autogen import UserProxyAgent, AssistantAgent, GroupChat, GroupChatManager, Agent

from .knowledge_extract_workflow import KnowledgeExtractionInternalWorkflow
from .knowledge_utils import search_kg, search_rag, search_web

from .knowledge_prompts import (
    EXTRACT_EXPERT_PROMPT,
    REPORT_EXPERT_PROMPT,
    KG_EXPERT_PROMPT,
    RAG_EXPERT_PROMPT,
    WEB_EXPERT_PROMPT,
    KNOWLEDGE_MANAGER_PROMPT
)

from configs import MAX_TRIES, SILENT

# Initialize module logger
logger = logging.getLogger(__name__)

class KnowledgeGroup:
    """
    Orchestrates a multi-agent group chat for Knowledge Extraction (Ingest) and Retrieval (QA).
    """

    def __init__(self, llm_config: Dict[str, Any], max_round: int):
        self.llm_config = llm_config

        # 1. Initialize Agents
        self.executor = self._create_executor()
        
        # Extraction Team
        self.extract_expert = self._create_assistant("extract_expert", EXTRACT_EXPERT_PROMPT)
        
        # Search Team
        self.report_expert = self._create_assistant("report_expert", REPORT_EXPERT_PROMPT)
        self.kg_expert = self._create_assistant("kg_expert", KG_EXPERT_PROMPT)
        self.rag_expert = self._create_assistant("rag_expert", RAG_EXPERT_PROMPT)
        self.web_expert = self._create_assistant("web_expert", WEB_EXPERT_PROMPT)

        # 2. Register capabilities
        self._register_tools()

        # 3. Initialize Orchestration with Custom Speaker Selection
        self.group_chat = GroupChat(
            agents=[
                self.executor,
                self.extract_expert,
                self.report_expert,
                self.kg_expert,
                self.rag_expert,
                self.web_expert
            ],
            messages=[],
            max_round=max_round,
            speaker_selection_method=self._custom_speaker_selection
        )

        self.manager = GroupChatManager(
            name="knowledge_manager",
            groupchat=self.group_chat,
            llm_config=self.llm_config,
            is_termination_msg=self._should_terminate,
            system_message=KNOWLEDGE_MANAGER_PROMPT,
            silent=SILENT
        )

    def _custom_speaker_selection(self, last_speaker: Agent, groupchat: GroupChat) -> Union[Agent, str, None]:
        messages = groupchat.messages
        if not messages:
            return "auto" 

        last_message = messages[-1]
        
        if last_speaker is self.executor:
            if len(messages) >= 2:
                caller_name = messages[-2].get("name")
                if caller_name == "kg_expert":
                    return self.kg_expert
                elif caller_name == "rag_expert":
                    return self.rag_expert
                elif caller_name == "web_expert":
                    return self.web_expert
                elif caller_name == "extract_expert":
                    return self.extract_expert
            return "auto" # Fallback

        if "tool_calls" in last_message or last_message.get("function_call"):
            return self.executor

        
        if "knowledge_executor" in last_speaker.name and "tool_calls" not in last_message:
            content = last_message.get("content", "").lower()
            if "extract" in content or ".pdf" in content or "抽取" in content:
                return self.extract_expert
            else:
                return self.kg_expert

        if last_speaker is self.extract_expert:
            return None 

        if last_speaker is self.kg_expert:
            return self.rag_expert

        if last_speaker is self.rag_expert:
            return self.web_expert

        if last_speaker is self.web_expert:
            return self.report_expert

        if last_speaker is self.report_expert:
            return None 

        return "auto"


    def get_manager(self) -> GroupChatManager:
        return self.manager

    def reset(self) -> None:
        self.executor.reset()
        self.extract_expert.reset()
        self.report_expert.reset() 
        self.kg_expert.reset()
        self.rag_expert.reset()
        self.web_expert.reset()
        self.group_chat.messages.clear()
        logger.info("Knowledge System memory reset successful.")

    def _create_executor(self) -> UserProxyAgent:
        return UserProxyAgent(
            name="knowledge_executor",
            human_input_mode="NEVER",
            code_execution_config={"use_docker": False},
            is_termination_msg=lambda x: "TERMINATE" in str(x.get("content", "")),
            silent=SILENT
        )

    def _create_assistant(self, name: str, system_message: str) -> AssistantAgent:
        return AssistantAgent(
            name=name,
            llm_config=self.llm_config,
            system_message=system_message,
            silent=SILENT
        )

    def _register_tools(self) -> None:
        self._extracting_engine = KnowledgeExtractionInternalWorkflow(self.llm_config)

        def extract_knowledge(input_path: str, output_dir: str):
            for attempt in range(MAX_TRIES):
                try:
                    print(f"[Info] Extraction attempt {attempt + 1} for {input_path}")
                    return self._extracting_engine.run(input_path, output_dir)
                except Exception as e:
                    print(e)

        autogen.register_function(
            extract_knowledge,
            caller=self.extract_expert,
            executor=self.executor,
            name="execute_extraction_workflow",
            description="Atomic workflow to extract knowledge from a PDF and save it."
        )

        autogen.register_function(
            search_kg,
            caller=self.kg_expert,
            executor=self.executor,
            name="search_kg",
            description="Search entities and relations in the Knowledge Graph (Neo4j)."
        )

        autogen.register_function(
            search_rag,
            caller=self.rag_expert,
            executor=self.executor,
            name="search_rag",
            description="Search relevant text chunks in the Vector Database."
        )

        autogen.register_function(
            search_web,
            caller=self.web_expert,
            executor=self.executor,
            name="search_web",
            description="Search the internet for real-time information."
        )

    def _should_terminate(self, msg: Dict[str, Any]) -> bool:
        content = str(msg.get("content", ""))
        sender = msg.get("name", "")

        if sender == "report_expert":
            return True 

        if sender == "knowledge_executor":
            if "Workflow Done" in content or "saved to" in content:
                return True

        return False