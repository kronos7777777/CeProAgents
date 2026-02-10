import os
import argparse
import copy
import autogen
# from configs import llm_config
from configs import GPT_MINI_CONFIG, GEMINI_MINI_CONFIG, CLAUDE_MINI_CONFIG, DEEPSEEK_MINI_CONFIG, QWEN_MINI_CONFIG
from CeProAgents.groups import ConceptGroup, clean_and_parse_json, save_json_file

def parse_arguments():
    parser = argparse.ArgumentParser(description="Batch process P&ID images using ConceptGroup agents.")
    parser.add_argument("--input_dir", type=str, default="D:\gitceagents\CE_Agents\CeProBench\CeProBench\concept\PID_image", 
                        help="Directory containing source images.")
    parser.add_argument("--output_dir", type=str, default="D:\\gitceagents\\CE_Agents\\results\\concept\\gemini-3-pro-preview\\parse", 
                        help="Directory to save output JSON files.")
    parser.add_argument("--start_id", type=int, default=1, help="Start image ID.")
    parser.add_argument("--end_id", type=int, default=116, help="End image ID (inclusive).")
    parser.add_argument("--model", type=str, default="gemini-3-pro-preview", 
                        help="LLM model type (e.g., gpt-4o, gpt-4o-mini). Overrides config.")
    return parser.parse_args()

def process_single_image_claude(concept_group, image_id, input_dir, output_dir):
    input_image_path = os.path.abspath(os.path.join(input_dir, f"{image_id}.png"))
    output_json_path = os.path.join(output_dir, f"{image_id}_parse_{args.model}.json")
    
    if not os.path.exists(input_image_path):
        print(f"[WARN] Skipping ID {image_id}: File not found at {input_image_path}")
        return

    print(f"[{image_id}] 🚀 Starting Parsing Phase for Claude...")

    root_user = autogen.UserProxyAgent(
        name="Root_User",
        human_input_mode="NEVER",  
        code_execution_config=False,
        max_consecutive_auto_reply=10,
    )
    concept_group.set_mode("parsing_only")
    concept_manager = concept_group.get_manager()
    
    task_payload = f"""
    **Task Type:** The Parsing Phase
    **Instruction:** Parse the P&ID image at the following path: "{input_image_path}"
    Call the parsing tool and return the JSON data.
    
    Example format (DO NOT return these specific values):
    ```json
    {{
      "equipments": [{{"identifier": "共沸剂冷凝器: E0301", "type" : "冷却/冷凝"}}], 
      "connections": [{{"source": "E0303", "target" : "E0401"}}]
    }}
    ```
    """
    try:
        root_user.initiate_chat(
            recipient=concept_manager,
            message=task_payload,
            summary_method=None,
        )
        real_content = None
        for msg in reversed(concept_group.groupchat.messages):
            if msg.get("name") == "concept_executor":
                real_content = msg.get("content")
                break

        if real_content:
            from ..ceproagents.groups import clean_and_parse_json, save_json_file
            pid_json = clean_and_parse_json(real_content)
            save_json_file(pid_json, output_json_path)
            print(f"[{image_id}] ✅ Success: {output_json_path}")
        else:
            print(f"[{image_id}] ❌ Error: No output from concept_executor.")

    except Exception as e:
        print(f"[{image_id}] ❌ Error: {str(e)}")
def process_single_image(concept_group, image_id, input_dir, output_dir):
    # Construct absolute path for tool execution safety
    input_image_path = os.path.abspath(os.path.join(input_dir, f"{image_id}.png"))
    output_json_path = os.path.join(output_dir, f"{image_id}_parse_{args.model}.json")
    
    if not os.path.exists(input_image_path):
        print(f"[WARN] Skipping ID {image_id}: File not found at {input_image_path}")
        return

    print(f"[{image_id}] Processing...")

    root_user = autogen.UserProxyAgent(
        name="Root_User",
        human_input_mode="NEVER",  
        code_execution_config=False,
        max_consecutive_auto_reply=1, 
    )
    
    concept_manager = concept_group.get_manager()
    
    # Explicitly trigger the 'Parsing Phase' protocol
    task_payload = f"""
    **Task Type:** The Parsing Phase
    **Instruction:** Parse the P&ID image at the following path: "{input_image_path}"
    Call the parsing tool and return the JSON data:
    Here is a simple example. Please note the formatting and the use of Chinese.
    ```json
        {{
        "equipments": [{{
            "identifier": "共沸剂冷凝器: E0301",
            "type" : "冷却/冷凝"
        }}], 
        "connections": [{{
            "source": "1,4-丁炔二醇冷却器: E0303",
            "target" : "去往E0401 BIT-SNYGWMD-0401"
        }}]
        }}
    ```.
    """

    try:
        chat_result = root_user.initiate_chat(
            recipient=concept_manager,
            message=task_payload,
            summary_method="reflection_with_llm", 
            summary_args={
                "summary_prompt": "Find the final JSON extracted from the image. Return ONLY the JSON code block."
            }
        )
        
        content = chat_result.summary
        if content:
            pid_json = clean_and_parse_json(content)
            if pid_json:
                save_json_file(pid_json, output_json_path)
                print(f"[{image_id}] ✅ Saved to {output_json_path}")
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
    # Apply model override
    # llm_config['config_list'][0]['model'] = args.model
    # Initialize Multi-Agent System
    concept_group = ConceptGroup(llm_config,current_mode="parsing_only")

    for i in range(args.start_id, args.end_id + 1):
        expected_filename = f"{i}_parse_{args.model}.json"
        expected_output_path = os.path.join(args.output_dir, expected_filename)
        
        if os.path.exists(expected_output_path):
            if os.path.getsize(expected_output_path) > 0:
                print(f"[{i}] ⏭️ Skipped: Result already exists ({expected_filename})")
                continue
            else:
                print(f"[{i}] ⚠️ Found empty file, re-processing...")
        concept_group.reset()
        process_single_image(concept_group, i, args.input_dir, args.output_dir)

if __name__ == "__main__":
    args = parse_arguments()

    main(args)
    