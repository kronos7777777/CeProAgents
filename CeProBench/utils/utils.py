import json
import re

def clean_and_parse_json(content: str):
    """
    Robust JSON Cleaner: Extracts JSON objects from LLM responses, 
    handling Markdown code blocks and extraneous text.
    """
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
    """Saves dictionary data to a JSON file with UTF-8 encoding."""
    try:
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=4)
        print(f"✅ Success! Standard JSON result saved to: {filename}")
    except Exception as e:
        print(f"❌ Save Failed: {e}")
        
def extract_json_from_markdown(text):
    """
    从混合文本中提取 JSON 代码块。
    支持 ```json ... ``` 格式或直接的 { ... } 格式。
    """
    if not text:
        return None

    # 1. 尝试匹配 Markdown 代码块 ```json ... ```
    pattern = r"```json\s*(.*?)\s*```"
    match = re.search(pattern, text, re.DOTALL)
    
    if match:
        json_str = match.group(1)
    else:
        # 2. 如果没有代码块，尝试寻找第一个 { 和最后一个 }
        # 这对于处理 LLM 偶尔忘记写 markdown 标记的情况很有用
        start = text.find('{')
        end = text.rfind('}')
        if start != -1 and end != -1:
            json_str = text[start:end+1]
        else:
            return None

    try:
        # 清理可能存在的注释或非标准字符
        return json.loads(json_str)
    except json.JSONDecodeError as e:
        print(f"Warning: JSON decoding failed: {e}")
        return None