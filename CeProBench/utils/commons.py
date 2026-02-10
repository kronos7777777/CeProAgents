import re
import json


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