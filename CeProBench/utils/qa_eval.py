import json
import re
import time
from openai import OpenAI
from configs import GOOGLE_API, GOOGLE_URL, GEMINI_MODEL

class QAEvaluator:
    def __init__(self, api_key=GOOGLE_API, base_url=GOOGLE_URL, model_name=GEMINI_MODEL):
        """
        Initializes the LLM Evaluator client.
        """
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model_name = model_name

    def _construct_prompt(self, question, ground_truth, prediction):

        return f"""
        You are an expert evaluator for Question Answering systems in the field of **Chemical Process Engineering**.
        Your task is to evaluate the quality of the 'Model Prediction' compared to the 'Ground Truth' based on the user's question regarding chemical processes, equipment, or operating conditions.

        **Evaluation Metrics (0-100 points each):**

        1.  **Correctness**: 
            - Factual accuracy regarding chemical reactions, process parameters (temperature, pressure), and equipment specifications.
            - Data precision and correct usage of units are strict requirements.
            - No fabrication of process steps.
        2.  **Rationality**: 
            - Logical consistency of the process flow description.
            - The explanation must follow a logical sequence (e.g., feed -> reaction -> separation).
            - No self-contradictions in thermodynamic or kinetic explanations.
        3.  **Clarity**: 
            - Conciseness and readability. 
            - Complex chemical terms should be used correctly but explained if necessary for clarity.
            - Avoid redundancy.
        4.  **Completeness**: 
            - Does it fully address the core question? 
            - Does it provide necessary details such as key operating conditions, catalysts, or specific equipment names mentioned in the Ground Truth?

        **Input Data:**
        [Question]: {question}
        [Ground Truth]: {ground_truth}
        [Model Prediction]: {prediction}

        **Output:** 
        Provide ONLY the JSON block following this structure:
        ```json
        {{
            "Correctness": <int>,
            "Rationality": <int>,
            "Clarity": <int>,
            "Completeness": <int>,
            "Critique": "<string, a concise summary of strengths and weaknesses>"
        }}
        ```
        """

    def _parse_response(self, content):
        """
        Robustly parses the JSON response from the LLM.
        """
        result = {
            "Correctness": 0, "Rationality": 0, "Clarity": 0, 
            "Completeness": 0, "Format": 0, 
            "Critique": "Parse Error"
        }
        
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            match = re.search(r"```json\s*(\{.*?\})\s*```", content, re.DOTALL)
            if not match:
                match = re.search(r"(\{.*\})", content, re.DOTALL)
            
            if match:
                try:
                    parsed = json.loads(match.group(1))
                    for k in result.keys():
                        if k in parsed:
                            result[k] = parsed[k]
                except:
                    print('[WARN] JSON Decode Error in regex extraction')
                    result["Critique"] = f"JSON Decode Error in regex extraction. Raw: {content[:50]}..."
            else:
                print('[WARN] No JSON found')
                result["Critique"] = f"No JSON found. Raw: {content[:50]}..."
                        
        return result

    def evaluate(self, question, ground_truth, prediction, retries=3):
        prompt = self._construct_prompt(question, ground_truth, prediction)
        
        for attempt in range(retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant that outputs strict JSON."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.0,
                )
                content = response.choices[0].message.content
                return self._parse_response(content)
            except Exception as e:
                print(f"[WARN] LLM attempt {attempt+1} failed: {e}")
                time.sleep(1)
        
        return {
            "Correctness": 0, "Rationality": 0, "Clarity": 0, 
            "Completeness": 0, "Format": 0, 
            "Critique": "API Call Failed after retries."
        }