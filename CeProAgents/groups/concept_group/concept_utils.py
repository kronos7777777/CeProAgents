import json
import re

def clean_and_parse_json(content: str):
    if not content: return {}
    content = str(content).strip()
    
    match = re.search(r"```(?:json)?\s*(.*?)\s*```", content, re.DOTALL)
    if match:
        content = match.group(1)
    
    if not content.startswith('{') and not content.startswith('['):
        match_brackets = re.search(r'(\{.*\}|\[.*\])', content, re.DOTALL)
        if match_brackets:
            content = match_brackets.group(1)

    try:
        return json.loads(content)
    except json.JSONDecodeError:
        return {} 

def save_json_file(data, filename):
    try:
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=4)
        print(f"✅ Success! Standard JSON result saved to: {filename}")
    except Exception as e:
        print(f"❌ Save Failed: {e}")
        
def extract_json_from_markdown(text):
    if not text:
        return None

    pattern = r"```json\s*(.*?)\s*```"
    match = re.search(pattern, text, re.DOTALL)
    
    if match:
        json_str = match.group(1)
    else:
        start = text.find('{')
        end = text.rfind('}')
        if start != -1 and end != -1:
            json_str = text[start:end+1]
        else:
            return None

    try:
        return json.loads(json_str)
    except json.JSONDecodeError as e:
        print(f"Warning: JSON decoding failed: {e}")
        return None

def get_final_pid_result(chat_result):
    if not chat_result or not chat_result.chat_history:
        print("Error: No chat history found.")
        return None

    print("\n===== Extracting Final JSON Result =====")
    
    for msg in reversed(chat_result.chat_history):
        content = msg.get('content', '')
        
        if content and "equipments" in content:
            json_data = extract_json_from_markdown(content)
            print(json_data)
            
            if json_data and isinstance(json_data, dict):
                if "equipments" in json_data:
                    
                    if "streams" in json_data and "connections" not in json_data:
                        print("Notice: Normalizing 'streams' to 'connections'...")
                        json_data["connections"] = json_data.pop("streams")
                    
                    print("Success: Valid PID JSON found.")
                    return json_data

    print("Error: No valid PID JSON structure found in the chat history.")
    return None