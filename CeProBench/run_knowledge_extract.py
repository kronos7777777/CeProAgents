import os
import argparse
import autogen
QWEN_MODEL = "qwen3-max"
from configs import GPT_MINI_CONFIG, GEMINI_MINI_CONFIG, CLAUDE_MINI_CONFIG, DEEPSEEK_MINI_CONFIG, QWEN_MINI_CONFIG
from CeProAgents import KnowledgeGroup
import logging

logging.getLogger("autogen").setLevel(logging.ERROR)

from configs import MAX_ROUND

def parse_arguments():
    parser = argparse.ArgumentParser(description="Batch process knowledge ingestion.")

    parser.add_argument("--input_dir", type=str, default="./CeProBench/knowledge/knowledge_raw", 
                        help="Directory containing source knowledge files.")
    
    parser.add_argument("--work_dir", type=str, default="./workdir/knowledge_extract", 
                        help="Root directory for output storage.")
    
    parser.add_argument("--start_id", type=int, default=1, help="Start ID.")
    parser.add_argument("--end_id", type=int, default=1, help="End ID (inclusive).")
    parser.add_argument("--model", type=str, default="gpt", help="LLM model used.")
    return parser.parse_args()

def get_source_file_path(input_dir, file_id):
    if os.path.exists(input_dir):
        prefix = f"{file_id}_"
        for fname in os.listdir(input_dir):
            if fname.startswith(prefix):
                return os.path.abspath(os.path.join(input_dir, fname))

    extensions = ['.txt', '.md', '.json', '.pdf']
    for ext in extensions:
        file_path = os.path.join(input_dir, f"{file_id}{ext}")
        if os.path.exists(file_path):
            return os.path.abspath(file_path)
            
    return None

def process_single_item(llm_config_local, file_id, input_dir, work_dir, model_name):
    input_file_path = get_source_file_path(input_dir, file_id)
    
    if not input_file_path:
        print(f"[WARN] ID {file_id}: No matching source file found in {input_dir}")
        return
    
    base_name = os.path.basename(input_file_path)
    dir_name = os.path.splitext(base_name)[0]
    
    log_path = os.path.join(work_dir, model_name, 'log.txt')
    current_output_dir = os.path.abspath(os.path.join(work_dir, model_name, dir_name))

    output_json_path = os.path.join(current_output_dir, f"{dir_name}_pred.json")

    if os.path.exists(output_json_path):
        print(f"[INFO] ID {file_id}: Output already exists at {output_json_path}, skipping.")
        return
    
    if not input_file_path:
        print(f"[WARN] ID {file_id}: No matching source file found in {input_dir}")
        return

    if not os.path.exists(current_output_dir):
        os.makedirs(current_output_dir)

    print(f"[{file_id}] Source: {os.path.basename(input_file_path)}")
    print(f"[{file_id}] Target: {current_output_dir}")

    knowledge_group = KnowledgeGroup(llm_config_local, MAX_ROUND)
    knowledge_agent = knowledge_group.get_manager()

    root_user = autogen.UserProxyAgent(
        name="Root_User",
        human_input_mode="NEVER",
        code_execution_config=False,
        max_consecutive_auto_reply=1
    )

    task_payload = f"""
    Command: INGEST
    
    Source File: "{input_file_path}"
    Target Directory: "{current_output_dir}"
    
    Instructions:
    1. Read the provided Source File.
    2. Process the content to build the knowledge base.
    3. Save the resulting index/data files to the Target Directory.
    """

    try:
        root_user.initiate_chat(
            knowledge_agent,
            message=task_payload,
        )
        with open(log_path, 'a', encoding='utf-8') as f:
            f.write(f"[{file_id}] Complete.\n")

    except Exception as e:
        with open(log_path, 'a', encoding='utf-8') as f:
            f.write(f"[{file_id}] Error: {str(e)}\n")

def main(args):
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

    if not os.path.exists(args.work_dir):
        os.makedirs(args.work_dir)

    print(f"=== Knowledge Ingestion Start ===")
    
    for i in range(args.start_id, args.end_id + 1):
        process_single_item(llm_config, i, args.input_dir, args.work_dir, args.model)

if __name__ == "__main__":
    args = parse_arguments()
    main(args)