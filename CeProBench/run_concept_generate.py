import os
import argparse
import glob
import json
import autogen
#from configs import llm_config
from configs import GPT_MINI_CONFIG, GEMINI_MINI_CONFIG, CLAUDE_MINI_CONFIG, DEEPSEEK_MINI_CONFIG, QWEN_MINI_CONFIG
from CeProAgents.groups import ConceptGroup, clean_and_parse_json, save_json_file

def parse_arguments():
    parser = argparse.ArgumentParser(description="Batch generate P&ID JSON data with metrics.")
    
    parser.add_argument("--input_dir", type=str, default="D:\gitceagents\CE_Agents\CeProBench\CeProBench\concept\PID_generate\case\case\huizong2", 
                        help="Directory containing source text prompts (.txt).")
    
    parser.add_argument("--output_dir", type=str, default="D:\\gitceagents\\CE_Agents\\results\\concept\\gemini-3-pro-preview\\generate_new_gemini", 
                        help="Directory to save output JSON files.")
    
    parser.add_argument("--model", type=str, default="gemini-3-pro-preview",        
                        help="LLM model type.")
    
    return parser.parse_args()

def process_single_prompt(concept_manager, user_proxy, file_path, output_dir, model_name):
    filename = os.path.basename(file_path)
    file_basename = os.path.splitext(filename)[0]
    
    path_final = os.path.join(output_dir, f"{file_basename}_final.json")
    path_first = os.path.join(output_dir, f"{file_basename}_first.json")
    path_stats = os.path.join(output_dir, f"{file_basename}_stats.json")
    
    print(f"[{file_basename}] Processing...")

    if not os.path.exists(file_path):
        print(f"[WARN] File not found: {file_path}")
        return

    with open(file_path, 'r', encoding='utf-8') as f:
        prompt_content = f.read().strip()

    if not prompt_content:
        print(f"[{file_basename}] ⚠️ Skipped: Empty prompt file.")
        return

    task_payload = f"""
    **Task Type:** P&ID Generation
    **Input Description:** 
    "{file_basename}"
    "{prompt_content}"

    **Instruction:** 
    Based on the description above, design the P&ID and output the result in structured JSON format.
    The JSON must adhere to the following schema(DO NOT return these specific values):
    ```json
    {{
      "equipments": [
        {{ "identifier": "甲醛炔化一级反应器: R0301", "type": "釜式反应器" }}
      ],
      "connections": [
        {{ "source": "乙炔自邻近厂经管道来", "target": "乙炔气压缩机: C0301" }}
      ]
    }}
    Here is a simple example. DO NOT return these specific values. Please note the formatting and the use of Chinese.
    """

    try:
        chat_result = user_proxy.initiate_chat(
            concept_manager,
            message=task_payload,
            summary_method=None, 
            summary_args={
                "summary_prompt": "Find the final JSON data generated based on the description. Return ONLY the JSON code block."
            }
        )
        
        first_draft_json = None
        last_generator_json = None
        iteration_count = 0

        full_history = concept_manager.groupchat.messages
        print(f"[{file_basename}] Chat history: {full_history}")
        for msg in full_history:
            sender_name = msg.get("name", "")
            content = msg.get("content", "")
            
            if sender_name == "generator_expert":
                parsed_json = clean_and_parse_json(content)
                
                if parsed_json:
                    iteration_count += 1

                    if first_draft_json is None:
                        first_draft_json = parsed_json
        for msg in reversed(full_history):
            if msg.get("name") == "generator_expert":
                parsed = clean_and_parse_json(msg.get("content", ""))
                if parsed:
                    last_generator_json = parsed
                    break 
        if first_draft_json:
            save_json_file(first_draft_json, path_first)
            print(f"  └─ [First Draft] Saved (Iterations so far: 1)")
        else:
            print(f"  └─ [First Draft] ❌ Not found.")
        final_content = last_generator_json
        if final_content:
            save_json_file(final_content, path_final)
            print(f"  └─ [Final Result] Saved.")
        else:
            print(f"  └─ [Final Result] ❌ No summary content.")
        stats_data = {
            "filename": file_basename,
            "model": model_name,
            "iteration_count": iteration_count,
            "has_first_draft": first_draft_json is not None,
            "has_final_result": final_content is not None
        }
        
        with open(path_stats, 'w', encoding='utf-8') as f:
            json.dump(stats_data, f, indent=4, ensure_ascii=False)
        print(f"  └─ [Stats] Saved. Total Iterations: {iteration_count}")

    except Exception as e:
        print(f"[{file_basename}] ❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()

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
    
    user_proxy = autogen.UserProxyAgent(
        name="User_Proxy",
        human_input_mode="NEVER",
        code_execution_config=False,
        max_consecutive_auto_reply=1,
    )
    concept_group = ConceptGroup(llm_config, current_mode="generation")
    concept_manager = concept_group.get_manager()

    # txt_files = glob.glob(os.path.join(args.input_dir, "*.txt"))
    
    # if not txt_files:
    #     print(f"No .txt files found in {args.input_dir}")
    #     return

    # print(f"Found {len(txt_files)} prompt files. Starting generation with metric tracking...")
    all_txt_files = glob.glob(os.path.join(args.input_dir, "*.txt"))
    if not all_txt_files:
        print(f"No .txt files found in {args.input_dir}")
        return

    existing_outputs = glob.glob(os.path.join(args.output_dir, "*_final.json"))
    processed_basenames = {os.path.basename(f).replace("_final.json", "") for f in existing_outputs}

    files_to_process = []
    for file_path in all_txt_files:
        file_basename = os.path.splitext(os.path.basename(file_path))[0]
        if file_basename in processed_basenames:
            # print(f"[{file_basename}] ⏭️ Skipped: Already exists in output directory.")
            continue
        files_to_process.append(file_path)

    total_all = len(all_txt_files)
    total_todo = len(files_to_process)
    total_done = total_all - total_todo

    print(f"Found {total_all} total files.")
    print(f"Already processed: {total_done} files.")
    print(f"Remaining to process: {total_todo} files.")
    print("-" * 50)

    if total_todo == 0:
        print("All files processed. Nothing to do.")
        return

    for file_path in sorted(files_to_process):
        concept_manager.reset()
        user_proxy.reset()
        process_single_prompt(concept_manager, user_proxy, file_path, args.output_dir, args.model)

if __name__ == "__main__":
    args = parse_arguments()
    main(args)