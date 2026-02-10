import os
import argparse
import glob
import autogen
#from configs import llm_config
from configs import GPT_MINI_CONFIG, GEMINI_MINI_CONFIG, CLAUDE_MINI_CONFIG, DEEPSEEK_MINI_CONFIG, QWEN_MINI_CONFIG
from CeProAgents.groups import ConceptGroup,clean_and_parse_json, save_json_file

def parse_arguments():
    parser = argparse.ArgumentParser(description="Batch process P&ID images for Completion/Inpainting.")
    
    parser.add_argument("--input_dir", type=str, default="D:\gitceagents\CE_Agents\CeProBench\CeProBench\concept\PID_complete", 
                        help="Directory containing MASKED source images.")
    
    parser.add_argument("--output_dir", type=str, default="D:\\gitceagents\\CE_Agents\\results\\concept\\gemini-3-pro-preview\\complete", 
                        help="Directory to save completed/reconstructed JSON files.")
    
    parser.add_argument("--start_id", type=int, default=1, help="Start image ID.")
    parser.add_argument("--end_id", type=int, default=1, help="End image ID (inclusive).")
    parser.add_argument("--model", type=str, default="gemini-3-pro-preview", 
                        help="LLM model type. Completion tasks usually require high reasoning capability.")
    return parser.parse_args()
def process_single_image_claude(concept_group, image_id, input_dir, output_dir):
    input_image_path = os.path.abspath(os.path.join(input_dir, f"{image_id}_mask_new.png"))
    output_json_path = os.path.join(output_dir, f"{image_id}_complete_{args.model}.json")
    
    if not os.path.exists(input_image_path):
        print(f"[WARN] Skipping ID {image_id}: File not found at {input_image_path}")
        return

    print(f"[{image_id}] 🚀 Starting Parsing Phase for Claude...")

    root_user = autogen.UserProxyAgent(
        name="Root_User",
        human_input_mode="NEVER",  
        code_execution_config=False,
        max_consecutive_auto_reply=1, 
    )
    
    concept_manager = concept_group.get_manager()
    
    task_payload = f"""
    **Task Type:** The Completion Phase
    
    **Instruction:** 
    The P&ID image at the following path contains a **masked area** (white box or occlusion): 
    "{input_image_path}"
    
    Your goal is to infer the missing process logic and reconstruct the P&ID structure.
    
    **Steps:**
    1. **Analyze Context:** Observe the pipelines and signals entering and exiting the masked area.
    2. **Infer Logic:** Based on chemical engineering principles, deduce what equipments or connections are missing (e.g., a pump usually follows a tank).
    DO NOT replace the word 'mask' in the equipments or connections. The output must be a Top 10 ranking of equipment types .
    DO NOT replace the word 'mask' in the equipments or connections. The output must be a Top 10 ranking of equipment types .
    DO NOT replace the word 'mask' in the equipments or connections. The output must be a Top 10 ranking of equipment types .
    The output must be a Top 10 ranking of equipment types .
    The output must be a Top 10 ranking of equipment types .
    The JSON must adhere to the following schema:Here is a simple example.The output must be a Top 10 ranking of equipment types .The output must be a Top 10 ranking of equipment types .The output must be a Top 10 ranking of equipment types .
    ```json
        {{
            "equipments": [...],
            "connections": [...],
            "completion": ["Most Likely Type",
            "2nd Most Likely",
            "3rd Most Likely",
            "4th Most Likely",
            "5th Most Likely",
            "6th Most Likely",
            "7th Most Likely",
            "8th Most Likely",
            "9th Most Likely",
            "10th Most Likely"]
            }}
            ```.
    
    """
    try:
        root_user.initiate_chat(
            recipient=concept_manager,
            message=task_payload,
            summary_method=None,
        )
        
        real_content = None
        for msg in reversed(concept_group.groupchat.messages):
            content = msg.get("content")
            if msg.get("name") == "completer_expert":
                if "completion" in content:
                    real_content = msg.get("content")
                    break

        if real_content:
            from CeProAgents.groups import clean_and_parse_json, save_json_file
            pid_json = clean_and_parse_json(real_content)
            save_json_file(pid_json, output_json_path)
            print(f"[{image_id}] ✅ Success: {output_json_path}")
        else:
            print(f"[{image_id}] ❌ Error: No output from concept_executor.")

    except Exception as e:
        print(f"[{image_id}] ❌ Error: {str(e)}")
def process_single_image(concept_group, image_id, input_dir, output_dir):
    input_image_path = os.path.abspath(os.path.join(input_dir, f"{image_id}_mask_new.png"))
    output_json_path = os.path.join(output_dir, f"{image_id}_complete_{args.model}.json")
    
    if not os.path.exists(input_image_path):
        print(f"[WARN] Skipping ID {image_id}: Masked file not found at {input_image_path}")
        return

    print(f"[{image_id}] Processing Completion Task...")

    root_user = autogen.UserProxyAgent(
        name="Root_User",
        human_input_mode="NEVER",  
        code_execution_config=False,
        max_consecutive_auto_reply=1, 
    )
    
    concept_manager = concept_group.get_manager()
    
    task_payload = f"""
    **Task Type:** The Completion Phase
    
    **Instruction:** 
    The P&ID image at the following path contains a **masked area** (white box or occlusion): 
    "{input_image_path}"
    
    Your goal is to infer the missing process logic and reconstruct the P&ID structure.
    
    **Steps:**
    1. **Analyze Context:** Observe the pipelines and signals entering and exiting the masked area.
    2. **Infer Logic:** Based on chemical engineering principles, deduce what equipments or connections are missing (e.g., a pump usually follows a tank).
    DO NOT replace the word 'mask' in the equipments or connections. The output must be a Top 10 ranking of equipment types .
    DO NOT replace the word 'mask' in the equipments or connections. The output must be a Top 10 ranking of equipment types .
    DO NOT replace the word 'mask' in the equipments or connections. The output must be a Top 10 ranking of equipment types .
    The output must be a Top 10 ranking of equipment types .
    The output must be a Top 10 ranking of equipment types .
    The JSON must adhere to the following schema:Here is a simple example.The output must be a Top 10 ranking of equipment types .The output must be a Top 10 ranking of equipment types .The output must be a Top 10 ranking of equipment types .
    ```json
        {{
            "equipments": [...],
            "connections": [...],
            "completion": ["Most Likely Type",
            "2nd Most Likely",
            "3rd Most Likely",
            "4th Most Likely",
            "5th Most Likely",
            "6th Most Likely",
            "7th Most Likely",
            "8th Most Likely",
            "9th Most Likely",
            "10th Most Likely"]
            }}
            ```.
    
    """

    try:
        chat_result = root_user.initiate_chat(
            concept_manager,
            message=task_payload,
            summary_method="reflection_with_llm", 
            summary_args={
                "summary_prompt": "Find the final Reconstructed JSON data representing the completed P&ID. Return ONLY the JSON code block."
            }
        )
        
        content = chat_result.summary
        if content:
            pid_json = clean_and_parse_json(content)
            if pid_json:
                save_json_file(pid_json, output_json_path)
                print(f"[{image_id}] ✅ Completion Saved to {output_json_path}")
            else:
                print(f"[{image_id}] ⚠️ JSON parsing failed.")
        else:
            print(f"[{image_id}] ❌ No response content.")

    except Exception as e:
        print(f"[{image_id}] ❌ Error: {str(e)}")

def main(args):    
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    if args.model == 'gpt':
        llm_config = GPT_MINI_CONFIG
    if args.model == 'gemini':
        llm_config = GEMINI_MINI_CONFIG
    if args.model == 'claude':
        llm_config = CLAUDE_MINI_CONFIG
    if args.model == 'deepseek':
        llm_config = DEEPSEEK_MINI_CONFIG
    if args.model == 'qwen':
        llm_config = QWEN_MINI_CONFIG

    # llm_config['config_list'][0]['model'] = args.model
    
    concept_group = ConceptGroup(llm_config,current_mode="completion")

    for i in range(args.start_id, args.end_id + 1):
        expected_filename = f"{i}_complete_{args.model}.json"
        expected_output_path = os.path.join(args.output_dir, expected_filename)
        
        if os.path.exists(expected_output_path):
            if os.path.getsize(expected_output_path) > 0:
                print(f"[{i}] ⏭️ Skipped: Result already exists ({expected_filename})")
                continue
            else:
                print(f"[{i}] ⚠️ Found empty file, re-processing...")
        process_single_image_claude(concept_group, i, args.input_dir, args.output_dir)

if __name__ == "__main__":
    args = parse_arguments()
    main(args)