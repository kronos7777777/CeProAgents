import autogen
from .concept_parse_workflow import execute_pid_parsing_tool
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

class ConceptGroup:
    # def custom_speaker_selection(self, last_speaker, groupchat):
        
    #     if last_speaker.name == "Root_User":
    #         return self.parser_expert

    #     if last_speaker.name == "parser_expert":
    #         return self.concept_executor

    #     if last_speaker.name == "concept_executor":

    #         last_msg = groupchat.messages[-1] if groupchat.messages else {}
    #         content = last_msg.get("content", "")
            
    #         content_str = str(content)
    #         # groupchat.messages[-1]["content"] = content_str

    #         if "equipments" in content_str and "connections" in content_str:
    #             if self.current_mode == "completion":
    #                 print(f"--- Transitioning to Completer Expert with {len(content_str)} chars of data ---")
    #                 print(f"--- Completer Expert content: {content_str} ---")
    #                 return self.completer_expert
    #             return None
    #         else:
    #             return self.concept_executor

    #     if last_speaker.name == "completer_expert":
    #         return None
    
    #     return None
    def __init__(self, llm_config,current_mode):
        self.current_mode = current_mode # : "parsing_only", "completion", "generation"
        
        # Creat Agent Experts
        self.concept_executor = autogen.UserProxyAgent(
            name="concept_executor",
            human_input_mode="NEVER",
            code_execution_config={"work_dir": "coding", "use_docker": False},
            is_termination_msg=lambda x: "TERMINATE" in str(x.get("content", "")),
            default_auto_reply="Tool execution complete. Reviewing results...", 
            system_message="""
            Executor. You execute the function calling code provided by the parser_expert.
            Do not analyze the data yourself. Just execute and return the raw output.
            """
        )

        # Parser Expert
        self.parser_expert = autogen.AssistantAgent(
            name="parser_expert",
            llm_config=llm_config,
            system_message=PARSR_EXPERT_PROMPT,
        )

        # Completer Expert
        self.completer_expert = autogen.AssistantAgent(
            name="completer_expert",
            llm_config=llm_config,
            system_message=COMPLETER_EXPERT_PROMPT,
        )

        # Corrector Expert
        self.corrector_expert = autogen.AssistantAgent(
            name="corrector_expert",
            llm_config=llm_config,
            default_auto_reply="Processing...",
            system_message=CORRECTOR_EXPERT_PROMPT,
        )

        # Generator Expert
        self.generator_expert = autogen.AssistantAgent(
            name="generator_expert",
            llm_config=llm_config,
            default_auto_reply="Processing...",
            system_message=GENERATOR_EXPERT_PROMPT,
        )

        # Register Tools
        autogen.register_function(
            execute_pid_parsing_tool,
            caller=self.parser_expert,
            executor=self.concept_executor,
            name="execute_pid_parsing_tool",
            description="Parses a PID image file and returns graph JSON."
        )

        # Create GroupChat
        self.groupchat = autogen.GroupChat(
            agents=[
                self.concept_executor, 
                self.parser_expert, 
                self.completer_expert, 
                self.corrector_expert, 
                self.generator_expert
            ],
            messages=[],
            max_round=12,
            # speaker_selection_method=self.custom_speaker_selection,
            allow_repeat_speaker=False
        )

        self.concept_manager = autogen.GroupChatManager(
            name="Concept_Manager",
            groupchat=self.groupchat, 
            llm_config=llm_config,
            default_auto_reply="Processing...",
            is_termination_msg=self._custom_termination_check,
            system_message=CONCEPT_MANAGER_PROMPT,
        )

    def set_mode(self, mode):
            self.current_mode = mode

    def get_manager(self):
        return self.concept_manager
    
    def reset(self):
        self.concept_executor.reset()
        self.parser_expert.reset()
        self.completer_expert.reset()
        self.corrector_expert.reset()
        self.generator_expert.reset()
        self.groupchat.messages = [] 
        self.groupchat.messages.clear()
        self.concept_manager.reset()
        print(">> System Memory Reset Successful.")
        
    def _custom_termination_check(self, msg):
        """
        Custom logic to determine if the GroupChat should stop based on the sender and content.
        This replaces the simple lambda check.
        """
        content = msg.get("content", "")
        sender = msg.get("name", "")

        # Condition A: Standard Termination (e.g., from Corrector)
        if "TERMINATE" in str(content):
            return True

        # Condition B: Parsing Phase Endpoint
        # If concept_executor speaks and provides a JSON block (indicating tool output is processed)
        if sender == "concept_executor" and self.current_mode == "parsing_only":
            # Check for JSON structure signature
            if "```json" in content or ("equipments" in content and "connections" in content):
                return True

        # Condition C: Completion Phase Endpoint
        # If completer_expert speaks, it's usually the final step of the chain
        if sender == "completer_expert" and self.current_mode == "completion":
            if "completion" in content:
                return True

        return False