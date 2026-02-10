import os
import json
import argparse
import logging
from typing import Dict, Any

import autogen

from CeProAgents.groups.parameter_group.aspen_utils import extract_aspen_block_set_parameters
from configs.llm_config import GEMINI_CONFIG,CLAUDE_CONFIG,DEEPSEEK_CONFIG,QWEN_CONFIG,GPT_CONFIG,GEMINI_MINI_CONFIG,GPT_MINI_CONFIG
from CeProAgents.groups.parameter_group import SimulationGroup
from CeProAgents.groups.parameter_group.simulation_utils import extract_aspen_flowsheet_connections

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def parse_arguments():
    parser = argparse.ArgumentParser(description="Batch run simulation parameter optimization.")
    
    parser.add_argument("--input_dir", type=str, 
                        default=r"D:\\simulation_group_marked_marked\\simulation_group\\casep\\casep\\simple",
                        help="Root directory containing subfolders with .bkp and .txt files.")
    
    parser.add_argument("--output_base_dir", type=str, 
                        default=r"D:\\simulation_group_marked_marked\\simulation_group\\results",
                        help="Base directory to save all task results.")
    
    return parser.parse_args()

def check_termination_and_log(msg: Dict[str, Any]) -> bool:
    if not msg: return False
    content = msg.get("content")
    content_str = str(content) if content else ""
    is_done = "TERMINATE" in content_str
    sender = msg.get("name", "Unknown")
    if is_done:
        print(f"\n[SYSTEM] 🛑 Termination detected from [{sender}].")
    return is_done

def process_single_case(bkp_path, goal_path, output_dir):
    case_name = os.path.splitext(os.path.basename(bkp_path))[0]
    logger.info(f"\n{'='*30}\n🚀 Starting Task: {case_name}\n{'='*30}")

    try:
        with open(goal_path, 'r', encoding='utf-8') as f:
            goal_content = f.read().strip()
        
        pid_data = extract_aspen_flowsheet_connections(bkp_path)
        param_data = extract_aspen_block_set_parameters(bkp_path)
    except Exception as e:
        logger.error(f"Failed to prepare data for {case_name}: {e}")
        return

    root_user = autogen.UserProxyAgent(
        name="Optimization_Starter",
        human_input_mode="NEVER",
        code_execution_config=False,
        llm_config=CLAUDE_CONFIG,
        max_consecutive_auto_reply=1,
        is_termination_msg=check_termination_and_log
    )

    global_context = {
        "bkp_file_path": bkp_path,
        "pid_json": pid_data,
        "optimization_goal": goal_content
    }
    sim_group = SimulationGroup(llm_config=CLAUDE_CONFIG, global_context=global_context)
    manager = sim_group.get_manager()

    task_payload = f"""
    **Mission**: Optimize simulation parameters for project: {case_name}
    **Context**:
    - bkp_file_path: "{bkp_path}"
    - optimization_goal: "{goal_content}"
    - pid_json: {json.dumps(pid_data, ensure_ascii=False)}
    - param_json: {json.dumps(param_data, ensure_ascii=False)}
    """
    try:
        chat_result = root_user.initiate_chat(
            manager,
            message=task_payload,
            summary_method="reflection_with_llm",
            summary_args={
                "summary_prompt": "Extract the FINAL optimized parameters and metrics as a clean JSON object."
            }
        )
        task_output_path = os.path.join(output_dir, case_name)
        os.makedirs(task_output_path, exist_ok=True)
        with open(os.path.join(task_output_path, "summary_result.json"), 'w', encoding='utf-8') as f:
            f.write(chat_result.summary)
        with open(os.path.join(task_output_path, "chat_history.json"), 'w', encoding='utf-8') as f:
            json.dump(chat_result.chat_history, f, indent=2, ensure_ascii=False)

        logger.info(f"✅ Task {case_name} completed. Results in {task_output_path}")

    except Exception as e:
        logger.error(f"❌ Task {case_name} failed: {e}")

def main():
    args = parse_arguments()
    input_root = os.path.abspath(args.input_dir)
    
    if not os.path.exists(input_root):
        logger.error(f"Input directory not found: {input_root}")
        return
    input_root = os.path.abspath(args.input_dir)
    output_root = os.path.abspath(args.output_base_dir)
    
    all_tasks = []
    for root, dirs, files in os.walk(input_root):
        for file in files:
            if file.endswith(".bkp"):
                base_name = os.path.splitext(file)[0]
                bkp_path = os.path.join(root, file)
                goal_path = os.path.join(root, base_name + ".txt")
                if os.path.exists(goal_path):
                    all_tasks.append((base_name, bkp_path, goal_path))

    tasks_to_run = []
    for case_name, bkp_p, goal_p in all_tasks:
        CURRENT_MODEL_NAME = "claude"
        expected_json_path = os.path.join(output_root, CURRENT_MODEL_NAME, f"{case_name}_Result_{CURRENT_MODEL_NAME}.json")
        
        if os.path.exists(expected_json_path):
            logger.info(f"⏭️  Skipping: {case_name} (Found {expected_json_path})")
        else:
            tasks_to_run.append((bkp_p, goal_p))

    logger.info(f"\n📊 Scan Finished: {len(all_tasks)} total, {len(all_tasks)-len(tasks_to_run)} skipped, {len(tasks_to_run)} to run.")
    # 遍历文件夹
    # tasks_found = []
    
    # for root, dirs, files in os.walk(input_root):
    #     # 寻找 bkp 文件
    #     for file in files:
    #         if file.endswith(".bkp"):
    #             bkp_path = os.path.join(root, file)
    #             # 寻找同名的 txt 文件
    #             base_name = os.path.splitext(file)[0]
    #             goal_file = base_name + ".txt"
    #             goal_path = os.path.join(root, goal_file)
                
    #             if os.path.exists(goal_path):
    #                 tasks_found.append((bkp_path, goal_path))
    #             else:
    #                 logger.warning(f"Found BKP but no matching TXT for: {file} in {root}")

    # if not tasks_found:
    #     logger.error("No valid BKP + TXT pairs found in the directory tree.")
    #     return

    # logger.info(f"Total tasks found: {len(tasks_found)}")

    for bkp_p, goal_p in tasks_to_run:
        process_single_case(bkp_p, goal_p, args.output_base_dir)

if __name__ == "__main__":
    main()