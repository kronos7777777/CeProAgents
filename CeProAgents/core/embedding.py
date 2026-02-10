import logging
from typing import List

from openai import OpenAI
from configs import OPENAI_API, EMBEDDING_MODEL, OPENAI_URL, MAX_TRIES

logger = logging.getLogger(__name__)

client = OpenAI(
    api_key=OPENAI_API,
    base_url=OPENAI_URL
)

def get_embedding(text: str, model: str = EMBEDDING_MODEL) -> List[float]:
    for attempt in range(MAX_TRIES):
        try:
            # Normalize text by removing newlines to prevent tokenization artifacts
            normalized_text = text.replace("\n", " ")
            
            response = client.embeddings.create(
                input=[normalized_text], 
                model=model
            )
            return response.data[0].embedding
        except Exception as e:
            logger.warning(f"Embedding attempt {attempt + 1} failed: {e}")
            if attempt == MAX_TRIES - 1:
                logger.error(f"All {MAX_TRIES} embedding attempts failed for text: {text}")
                return []