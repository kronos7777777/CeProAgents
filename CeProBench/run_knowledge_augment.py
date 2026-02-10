import os
import argparse
import json
import logging
import autogen
from configs import GPT_CONFIG, GEMINI_CONFIG, CLAUDE_CONFIG, DEEPSEEK_MINI_CONFIG, QWEN_CONFIG
from CeProAgents import KnowledgeGroup
from configs import MAX_ROUND

logging.getLogger("autogen").setLevel(logging.ERROR)

def parse_arguments():
    parser = argparse.ArgumentParser(description="Batch process QA without intermediate file storage.")
    
    parser.add_argument("--input_file", type=str, 
                        default="./CeProBench/knowledge/knowledge_agument/question.jsonl", 
                        help="Path to the source question.jsonl file.")
    
    parser.add_argument("--work_dir", type=str, default="./workdir/knowledge_augment", 
                        help="Root directory for output storage.")
    
    parser.add_argument("--start_id", type=int, default=1, help="Start ID.")
    parser.add_argument("--end_id", type=int, default=1, help="End ID (inclusive).")
    parser.add_argument("--model", type=str, default="gpt", help="LLM model used.")
    return parser.parse_args()

def load_data_map(input_file):
    data_map = {}
    if not os.path.exists(input_file):
        print(f"[ERROR] Input file not found: {input_file}")
        return data_map

    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line: continue
                try:
                    item = json.loads(line)
                    if 'id' in item:
                        data_map[int(item['id'])] = item
                except json.JSONDecodeError:
                    continue
    except Exception as e:
        print(f"[ERROR] Reading jsonl file failed: {e}")
    return data_map

def get_agent_response(chat_result):
    if not chat_result:
        return ""
    
    try:
        if hasattr(chat_result, 'summary') and chat_result.summary:
            return chat_result.summary
        
        if hasattr(chat_result, 'chat_history'):
            for message in reversed(chat_result.chat_history):
                content = message.get('content', '')
                if content and isinstance(content, str) and "TERMINATE" not in content:
                    return content
                
    except Exception as e:
        print(f"[WARN] Failed to extract message: {e}")
    
    return "No response captured."

def process_single_item(llm_config_local, item, output_file, model_name):
    item_id = int(item['id'])
    question_class = item.get('class', 'General')
    question_text = item.get('question', '')
    ground_truth = item.get('answer', '')

    if os.path.exists(output_file):
        with open(output_file, 'r', encoding='utf-8') as f:
            for line in f:
                existing_item = json.loads(line)
                if existing_item.get('id') == item_id:
                    print(f"[INFO] ID {item_id}: Output already exists at {output_file}, skipping.")
                    return

    print(f"[{item_id}] Processing Question: {question_text[:50]}...")

    knowledge_group = KnowledgeGroup(llm_config_local, MAX_ROUND)
    knowledge_agent = knowledge_group.get_manager()

    root_user = autogen.UserProxyAgent(
        name="Root_User",
        human_input_mode="NEVER",
        code_execution_config=False,
        max_consecutive_auto_reply=1
    )

    task_payload = f"""
    Command: Query Task
    
    Context:
    - Query: "{question_text}"
    
    Instructions:
    Knowledge Augment(KG, RAG and Web) based on this query and then give the final answer.
    """

    prediction = ""
    try:
        chat_result = root_user.initiate_chat(
            knowledge_agent,
            message=task_payload,
            summary_method="reflection_with_llm", 
            summary_args={
                "summary_prompt": "Output the final report"}
        )
        prediction = chat_result.summary
        
    except Exception as e:
        print(f"[{item_id}] Error: {str(e)}")
        prediction = f"Error: {str(e)}"

    result_item = {
        "id": item_id,
        "class": question_class,
        "question": question_text,
        "ground_truth": ground_truth,
        "prediction": prediction
    }

    with open(output_file, 'a', encoding='utf-8') as f:
        f.write(json.dumps(result_item, ensure_ascii=False) + "\n")

def main(args):
    if args.model == 'gpt':
        llm_config = GPT_CONFIG
    elif args.model == 'gemini':
        llm_config = GEMINI_CONFIG
    elif args.model == 'claude':
        llm_config = CLAUDE_CONFIG
    elif args.model == 'deepseek':
        llm_config = DEEPSEEK_MINI_CONFIG
    if args.model == 'qwen':
        llm_config = QWEN_CONFIG

    output_dir = os.path.join(args.work_dir, args.model)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    output_file = os.path.join(output_dir, "answers_wo_KB.jsonl")
    print(f"=== QA Start ===")
    print(f"Results will be saved to: {output_file}")
    
    data_map = load_data_map(args.input_file)
    if not data_map:
        return

    for i in range(args.start_id, args.end_id + 1):
        if i in data_map:
            process_single_item(llm_config, data_map[i], output_file, args.model)
        else:
            print(f"[WARN] ID {i} not found.")

if __name__ == "__main__":
    args = parse_arguments()
    main(args)