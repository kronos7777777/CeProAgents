import json
import re
import logging
import ast
import copy
from typing import Dict, Any, List, Optional

import autogen
from simulation_utils import run_aspen_with_structured_io
from simulation_prompts import (
    INTERNAL_OPTIMIZER_PROMPT,
    INTERNAL_ANALYST_PROMPT 
)

logger = logging.getLogger(__name__)

class ParameterOptimizationWorkflow:
    """
    Internal Workflow Engine: Optimize Aspen simulation parameters.
    Fully captures iteration history (inputs, outputs, reasoning).
    """

    MAX_ITERATIONS = 20
    TERMINATION_KEYWORD = "TERMINATE"

    def __init__(self, llm_config: Dict[str, Any]):
        self.context: Dict[str, Any] = {
            "bkp_file_path": "",
            "pid_json": {},
            "ranges": {},
            "input_config": {},
            "output_config": {},
            "iteration_history": [], 
            "optimization_goal": "" 
        }
        
        self.internal_admin = autogen.UserProxyAgent(
            name="Inner_Admin",
            human_input_mode="NEVER",
            code_execution_config=False,
            max_consecutive_auto_reply=0
        )

        self.analyst = autogen.AssistantAgent(
            name="Inner_Analyst",
            llm_config=llm_config,
            system_message=INTERNAL_ANALYST_PROMPT
        )

        self.optimizer = autogen.AssistantAgent(
            name="Inner_Optimizer",
            llm_config=llm_config,
            system_message=INTERNAL_OPTIMIZER_PROMPT
        )

    def _convert_keys_to_strings(self, result: Dict) -> Dict:
        """Helper: Converts tuple keys in results back to strings for JSON serialization."""
        new_result = {}
        for k, v in result.items():
            new_key = str(k) if isinstance(k, tuple) else k
            if isinstance(v, dict):
                new_result[new_key] = self._convert_keys_to_strings(v)
            else:
                new_result[new_key] = v
        return new_result

    def _extract_json_payload(self, content: str) -> Dict[str, Any]:
        if not content: return {}
        try: return json.loads(content)
        except json.JSONDecodeError:
            match = re.search(r"```json\s*(\{.*?\})\s*```", content, re.DOTALL)
            if match: return json.loads(match.group(1))
            match = re.search(r"(\{.*\})", content, re.DOTALL)
            if match: return json.loads(match.group(1))
            return {}

    def _standardize_input_format(self, config: Dict) -> Dict:
        """Standardize LLM inputs to List[Tuple] format for the tool."""
        new_config = {}
        for k, v in config.items():
            new_key = k
            if isinstance(k, str) and k.startswith("(") and k.endswith(")"):
                try:
                    val = ast.literal_eval(k)
                    if isinstance(val, tuple): new_key = val
                except: pass 
            
            if isinstance(new_key, str): # Block ID
                if isinstance(v, list):
                    cleaned_list = []
                    for item in v:
                        if isinstance(item, (list, tuple)): cleaned_list.append(tuple(item))
                        else: cleaned_list.append(item)
                    new_config[new_key] = cleaned_list
                elif isinstance(v, dict):
                    new_config[new_key] = list(v.items())
                else:
                    new_config[new_key] = v
            else:
                new_config[new_key] = v
        return new_config

    def _create_analysis_message(self, sim_result_str: str, current_iter: int) -> str:
        history_snippet = self.context['iteration_history'][-3:] 
        goal = self.context.get('optimization_goal', "Improve efficiency.")
        current_cfg_str = json.dumps(self.context['input_config'], default=str)

        return (
            f"[Current Status]\n"
            f"Iteration: {current_iter} / {self.MAX_ITERATIONS}\n\n"
            f"[PiD Structure]\n{json.dumps(self.context['pid_json'], indent=2)}\n\n"
            f"[Current Configuration]\n{current_cfg_str}\n\n"
            f"[Simulation Output]\n{sim_result_str}\n\n"
            f"[Optimization Goal (STRICT)]\n{goal}\n\n"
            f"[History (Last 3)]\n{json.dumps(history_snippet, default=str)}\n\n"
            f"[Task]\n"
            f"Compare results against the goal. Decide to '{self.TERMINATION_KEYWORD}' or continue."
        )

    def _create_optimization_message(self, analysis_result: str) -> str:
        current_cfg_str = json.dumps(self.context['input_config'], default=str)
        return (
            f"[Analyst Feedback]\n{analysis_result}\n\n"
            f"[PiD Structure]\n{json.dumps(self.context['pid_json'], indent=2)}\n\n"
            f"[Current Config]\n{current_cfg_str}\n\n"
            f"[Ranges/Constraints]\n{json.dumps(self.context['ranges'])}\n\n"
            f"[Task]\n"
            f"Optimize the 'Current Config'. Return the FULL updated configuration JSON.\n"
            f"Key format requirement: For Blocks, values must be lists of pairs. e.g. [['PARAM', val]]."
        )


    def _execute_simulation(self) -> str:
        """Executes simulation and returns JSON string."""
        try:
            raw_input_config = self.context['input_config']
            processed_input = self._standardize_input_format(raw_input_config)
            processed_output_conf = self._standardize_input_format(self.context['output_config'])
            file_path = self.context['bkp_file_path']

            results = run_aspen_with_structured_io(
                aspen_file_path=file_path,
                input_config=processed_input,
                output_config=processed_output_conf,
                visible=False,
                max_wait_time=300
            )

            serializable_results = self._convert_keys_to_strings(results)
            return json.dumps(serializable_results, ensure_ascii=False)
        except Exception as e:
            logger.error(f"Simulation execution failed: {e}", exc_info=True)
            return json.dumps({"error": str(e)}, ensure_ascii=False)

    def run(self, 
            bkp_file_path: str, 
            pid_json: dict, 
            input_config: dict, 
            output_config: dict, 
            parameters_ranges: dict,
            optimization_goal: str 
            ) -> str:
        
        logger.info("Starting Internal Optimization Workflow...")
        
        self.context.update({
            "bkp_file_path": bkp_file_path,
            "pid_json": pid_json,
            "input_config": input_config,
            "output_config": output_config,
            "ranges": parameters_ranges,
            "iteration_history": [], 
            "optimization_goal": optimization_goal
        })

        iteration = 0
        final_verdict = "Max iterations reached."

        while iteration < self.MAX_ITERATIONS:
            iteration += 1
            logger.info(f"--- Iteration {iteration} ---")

            sim_result_str = self._execute_simulation()
            
            try:
                sim_result_data = json.loads(sim_result_str)
            except:
                sim_result_data = sim_result_str

            current_history_entry = {
                "iteration": iteration,
                "input_parameters": copy.deepcopy(self.context["input_config"]),
                "simulation_results": sim_result_data,
                "analyst_feedback": None 
            }
            self.context["iteration_history"].append(current_history_entry)

            self.internal_admin.initiate_chat(
                self.analyst,
                message=self._create_analysis_message(sim_result_str, iteration),
                max_turns=1
            )
            analysis_result = self.internal_admin.last_message(self.analyst)["content"]
            
            self.context["iteration_history"][-1]["analyst_feedback"] = analysis_result
            logger.info(f"Analyst Verdict: {analysis_result}")
            verdict_upper = analysis_result.strip()

            is_negative_context = ("NOT TERMINATE" in verdict_upper) or ("DO NOT TERMINATE" in verdict_upper)

            has_termination_keyword = self.TERMINATION_KEYWORD in verdict_upper

            if has_termination_keyword and not is_negative_context:
                logger.info(f"Optimization converged at iteration {iteration}.")
                final_verdict = analysis_result
                break
            elif has_termination_keyword and is_negative_context:
                logger.info(f"False Alarm: Analyst said '{self.TERMINATION_KEYWORD}' but with negation. Continuing...")

            self.internal_admin.initiate_chat(
                self.optimizer,
                message=self._create_optimization_message(analysis_result),
                max_turns=1
            )
            opt_response = self.internal_admin.last_message(self.optimizer)["content"]
            
            payload = self._extract_json_payload(opt_response)
            new_config = payload.get("input_config", payload)

            if new_config and isinstance(new_config, dict) and len(new_config) > 0:
                self.context["input_config"] = new_config

                logger.info("Configuration updated by Optimizer.")
            else:
                logger.error("Optimizer failed to generate valid config.")
                final_verdict = "Optimizer Failure"
                break

        return json.dumps({
            "final_config": self.context["input_config"],
            "analyst_conclusion": final_verdict,
            "total_iterations": iteration,
            "full_history": self.context["iteration_history"] 
        }, ensure_ascii=False)