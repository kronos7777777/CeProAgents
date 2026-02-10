import logging
from typing import Dict, Any, List
import os
import time
import autogen
import ast 
from typing import Any
import json

from autogen import UserProxyAgent, AssistantAgent, GroupChat, GroupChatManager

from aspen_utils import extract_aspen_block_set_parameters
from simulation_optimize_workflow import ParameterOptimizationWorkflow
from simulation_prompts import (
    SENTINEL_EXPERT_PROMPT,
    CONFIGER_EXPERT_PROMPT,
    INITIALIZER_EXPERT_PROMPT,
    OPTIMIZER_EXPERT_PROMPT,
    SIMULATION_MANGER_PROMPT
)

# Initialize module logger
logger = logging.getLogger(__name__)

def fix_tuple_keys(obj):
    if isinstance(obj, dict):
        new_dict = {}
        for k, v in obj.items():
            new_key = f"{k[0]},{k[1]}" if isinstance(k, tuple) else k
            new_dict[new_key] = fix_tuple_keys(v)
        return new_dict
    elif isinstance(obj, list):
        return [fix_tuple_keys(i) for i in obj]
    else:
        return obj
class SimulationGroup:
    def custom_speaker_selection(self, last_speaker, groupchat):

        print(f"\n[DEBUG] Current Speaker: {last_speaker.name}")
        if groupchat.messages:
            print(f"[DEBUG] Last Message Preview: {groupchat.messages[-1]['content'][:50]}...")

        messages = groupchat.messages

        if last_speaker.name == "Optimization_Starter":
            return self.sentinel

        if last_speaker.name == "sentinel_expert":
            return self.configer

        if last_speaker.name == "configer_expert":
            return self.initializer

        if last_speaker.name == "initialize_expert":
            return self.optimizer

        if last_speaker.name == "optimize_expert":
            return self.executor

        if last_speaker.name == "executor": 
            return None

        return None

    def __init__(self, llm_config: Dict[str, Any], global_context: Dict[str, Any] = None):
        self.llm_config = llm_config
        self.global_context = global_context

        # Initialize Agents
        self.executor = self._create_executor()
        self.sentinel = self._create_assistant("sentinel_expert", SENTINEL_EXPERT_PROMPT)
        self.configer = self._create_assistant("configer_expert", CONFIGER_EXPERT_PROMPT)
        # # self.chemist = self._create_assistant("chemist_expert", CHEMIST_EXPERT_PROMPT)
        # self.inspector = self._create_assistant("inspector_expert", INSPECTOR_EXPERT_PROMPT)
        self.initializer = self._create_assistant("initialize_expert", INITIALIZER_EXPERT_PROMPT)
        self.optimizer = self._create_assistant("optimize_expert", OPTIMIZER_EXPERT_PROMPT)

        # Register capabilities
        self._register_tools()

        # Initialize Orchestration
        self.group_chat = GroupChat(
            agents=[
                self.executor,
                self.sentinel,
                self.configer,
                self.initializer,
                self.optimizer
            ],
            messages=[],
            max_round=15,
            speaker_selection_method=self.custom_speaker_selection,
            allow_repeat_speaker=False
        )

        self.manager = GroupChatManager(
            name="simulation_manager",
            groupchat=self.group_chat,
            llm_config=False,
            is_termination_msg=self._should_terminate,
            system_message=SIMULATION_MANGER_PROMPT
        )

    def get_manager(self) -> GroupChatManager:
        return self.manager

    def reset(self) -> None:
        agents = [
            self.executor, self.sentinel, self.configer,
            self.initializer, self.optimizer
        ]
        for agent in agents:
            agent.reset()
        self.group_chat.messages.clear()
        logger.info("Simulation group memory reset successful.")

    def _create_executor(self) -> UserProxyAgent:
        return UserProxyAgent(
            name="executor",
            human_input_mode="NEVER",
            code_execution_config={"work_dir": "coding", "use_docker": False},
            is_termination_msg=lambda x: "TERMINATE" in str(x.get("content", "")),
        )

    def _create_assistant(self, name: str, system_message: str) -> AssistantAgent:
        return AssistantAgent(
            name=name,
            llm_config=self.llm_config,
            system_message=system_message
        )

    def _register_tools(self) -> None:
        import re
        import json

        _simulating_engine = ParameterOptimizationWorkflow(self.llm_config)
        
        def run_optimization_wrapper(
                bkp_file_path: str = None,
                pid_json: Any = None,
                input_config: Any = None,
                output_config: Any = None,
                parameters_ranges: Any = None,
                optimization_goal: str = None
        ) -> str:
            print(f"\n[Wrapper] Tool called. Checking arguments...")

            input_config = None
            output_config = None
            parameters_ranges = None
            print("[Wrapper] Missing config dicts. Searching 'initialize_expert' history...")
            full_chat_history = self.group_chat.messages
            for msg in self.group_chat.messages:
                history_item = {
                    "role": msg.get("name") or msg.get("role"),
                    "content": msg.get("content", ""),
                    "tool_calls": msg.get("tool_calls", None) 
                    }
                full_chat_history.append(history_item)
                history_save_path = os.path.join("D:\\simulation_group_marked_marked\\simulation_group\\results\\claude", f"caseclaude_Chat_History.json")
                try:
                    serializable_history = fix_tuple_keys(full_chat_history)
                    with open(history_save_path, 'w', encoding='utf-8') as f:
                        json.dump(serializable_history, f, indent=4, ensure_ascii=False)
                    print(f"[Wrapper] 💾 Expert conversation history saved to: {history_save_path}")
                except Exception as e:
                    print(f"[Wrapper] ❌ Failed to save chat history: {e}")
            found_initializer = False
            for msg in reversed(full_chat_history):
                if msg['name'] == 'initialize_expert':
                    content = msg.get('content', '')
                    try:
                        code_match = re.search(r"```(?:json|python)?(.*?)```", content, re.DOTALL)
                        if code_match:
                            code_str = code_match.group(1).strip()
                        else:
                            code_str = content.strip()

                        data = ast.literal_eval(code_str)
                        if isinstance(data, dict):
                            if input_config is None: input_config = data.get("input_config")
                            if output_config is None: output_config = data.get("output_config")
                            if parameters_ranges is None: parameters_ranges = data.get("parameters_ranges")
                            found_initializer = True
                            print("[Wrapper] Successfully extracted configs from history.")
                            break
                    except Exception as e:
                        print(f"[Wrapper] Failed to parse Initializer output: {e}")

            if not found_initializer:
                print("[Wrapper] Warning: Could not find valid output from initialize_expert.")

            missing_params = []
            if input_config is None: missing_params.append("input_config")
            if optimization_goal is None: missing_params.append("optimization_goal")

            if missing_params:
                error_msg = f"Error: Failed to auto-extract parameters: {', '.join(missing_params)}. Please ensure previous agents generated valid data."
                print(f"[Wrapper] {error_msg}")
                return error_msg
            if output_config:
                forbidden_keys = ["ANNUAL_OP_COST", "TAC", "COST", "PRICE", "OP_COST", "PROFIT"]
                cleaned_output_config = {}

                for key, val in output_config.items():
                    val_str = str(val).upper()

                    if any(bad_word in val_str for bad_word in forbidden_keys):
                        print(f"[Wrapper] Detected forbidden metric '{val}' for block '{key}'. Auto-correcting...")

                        key_str = str(key).upper()

                        if key_str.startswith("T"):
                            cleaned_output_config[key] = "DUTY"
                            print(f"   -> Auto-corrected to 'DUTY' (Column detected: {key})")
                        else:
                            cleaned_output_config[key] = "QCALC"
                            print(f"   -> Auto-corrected to 'QCALC' (Reactor/Heater detected: {key})")

                    else:
                        cleaned_output_config[key] = val

                output_config = cleaned_output_config
            print("[Wrapper] All parameters ready. Executing engine...")
            result = _simulating_engine.run(
                bkp_file_path=self.global_context.get("bkp_file_path"),
                pid_json=self.global_context.get("pid_json"),
                input_config=input_config,
                output_config=output_config,
                parameters_ranges=parameters_ranges,
                optimization_goal=self.global_context.get("optimization_goal")
            )
            bkp_name = os.path.basename(self.global_context.get("bkp_file_path", "unknown")).replace(".bkp", "")
            print(f"[Wrapper] bkp_name: {bkp_name}")
            save_filename = f"{bkp_name}_Result_claude.json"
            save_path = os.path.join("D:\\simulation_group_marked_marked\\simulation_group\\results\\claude", save_filename) # 建议指定固定目录   
            
            try:
                result_data = json.loads(result)
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                with open(save_path, 'w', encoding='utf-8') as f:
                    json.dump(result_data, f, indent=4, ensure_ascii=False)
                print(f"[Wrapper] ✅ Full optimization history saved to: {save_path}")
            except Exception as e:
                print(f"[Wrapper] ❌ Failed to save local file: {e}")
            return f"***** Simulation Result *****\n{result}\n\nTERMINATE"

        autogen.register_function(
            run_optimization_wrapper,
            caller=self.optimizer,
            executor=self.executor,
            name="optimize_parameter_workflow",
            description="Optimizes simulation parameters based on input configuration JSON."
        )

        # autogen.register_function(
        #     run_optimization_wrapper,
        #     caller=self.optimizer,
        #     executor=self.executor,
        #     name="optimize_parameter_workflow",
        #     description="Optimizes simulation parameters based on input configuration JSON."
        # )

        # autogen.register_function(
        #     extract_aspen_block_set_parameters,
        #     caller=self.configer,
        #     executor=self.executor,
        #     name="extract_aspen_block_set_parameters",
        #     description="Extract "
        # )

    def _should_terminate(self, msg: Dict[str, Any]) -> bool:
        """
        Custom termination logic for the group chat.
        Checks for explicit 'TERMINATE' signal.
        """
        content = str(msg.get("content", ""))
        print(f"[Wrapper] Checking termination signal: {content}")
        return "TERMINATE" in content
