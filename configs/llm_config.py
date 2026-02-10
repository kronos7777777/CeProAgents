# =========================
# General Settings
# =================
TIMEOUT = 300
TEMPERATURE = 0.0
MAX_TRIES = 2

# =========================
# URL Settings
# =========================
DEFAULT_URL = ""
OPENAI_URL = DEFAULT_URL
GOOGLE_URL = DEFAULT_URL
ANTHROPIC_URL = DEFAULT_URL
DEEPSEEK_URL = DEFAULT_URL
QWEN_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"

# ========================
# API Settings
# =========================
DEFAULT_API = ""
OPENAI_API = DEFAULT_API
GOOGLE_API = DEFAULT_API
ANTHROPIC_API = DEFAULT_API
DEEPSEEK_API = DEFAULT_API
Qwen_API = ""

# =========================
# Model Settings
# =========================
EMBEDDING_MODEL = "text-embedding-3-small"

GPT_MODEL = "gpt-5-2025-08-07"
GPT_MINI_MODEL = "gpt-5-mini-2025-08-07"

GEMINI_MODEL = "gemini-3-pro-preview"
GEMINI_MINI_MODEL = "gemini-3-flash-preview"

CLAUDE_MODEL = "claude-opus-4-5-20251101"
CLAUDE_MINI_MODEL = "claude-haiku-4-5-20251001"

DEEPSEEK_MODEL = "deepseek-reasoner"
DEEPSEEK_MINI_MODEL = "deepseek-chat"

QWEN_MINI_MODEL = "qwen3-vl-plus"
QWEN_MODEL = "qwen3-max"

UNIFIED_EXTRACTOR_MODELS = [GPT_MODEL, GEMINI_MODEL, CLAUDE_MODEL, GPT_MINI_MODEL, GEMINI_MINI_MODEL, CLAUDE_MINI_MODEL, QWEN_MINI_MODEL]

# =========================
# LLM Configurations
# =========================

# DeepSeek
DEEPSEEK_CONFIG = {
    "config_list": [
        {"model": DEEPSEEK_MODEL, "api_key": DEEPSEEK_API, "base_url": DEEPSEEK_URL}
    ],
    "temperature": TEMPERATURE,
    "timeout": TIMEOUT,
}
DEEPSEEK_MINI_CONFIG = {
    "config_list": [
        {"model": DEEPSEEK_MINI_MODEL, "api_key": DEEPSEEK_API, "base_url": DEEPSEEK_URL}
    ],
    "temperature": TEMPERATURE,
    "timeout": TIMEOUT,
}


# GPT (OpenAI)
GPT_CONFIG = {
    "config_list": [
        {"model": GPT_MODEL, "api_key": OPENAI_API, "base_url": OPENAI_URL}
    ],
    "temperature": TEMPERATURE,
    "timeout": TIMEOUT,
}
GPT_MINI_CONFIG = {
    "config_list": [
        {"model": GPT_MINI_MODEL, "api_key": OPENAI_API, "base_url": OPENAI_URL}
    ],
    "temperature": TEMPERATURE,
    "timeout": TIMEOUT,
}

# Gemini (Google)
GEMINI_CONFIG = {
    "config_list": [
        {"model": GEMINI_MODEL, "api_key": GOOGLE_API, "base_url":  GOOGLE_URL}
    ],
    "temperature": TEMPERATURE,
    "timeout": TIMEOUT,
}
GEMINI_MINI_CONFIG = {
    "config_list": [
        {"model": GEMINI_MINI_MODEL, "api_key": GOOGLE_API, "base_url":  GOOGLE_URL}
    ],
    "temperature": TEMPERATURE,
    "timeout": TIMEOUT,
}

# Claude (Anthropic)
CLAUDE_CONFIG = {
    "config_list": [
        {"model": CLAUDE_MODEL, "api_key": ANTHROPIC_API, "base_url":  ANTHROPIC_URL}
    ],
    "temperature": TEMPERATURE,
    "timeout": TIMEOUT,
}

CLAUDE_MINI_CONFIG = {
    "config_list": [
        {"model": CLAUDE_MINI_MODEL, "api_key": ANTHROPIC_API, "base_url":  ANTHROPIC_URL}
    ],
    "temperature": TEMPERATURE,
    "timeout": TIMEOUT,
}

QWEN_MINI_CONFIG = {
    "config_list": [
        {"model": QWEN_MINI_MODEL, "api_key": Qwen_API, "base_url":  QWEN_URL}
    ],
    "temperature": TEMPERATURE,
    "timeout": TIMEOUT,
}

QWEN_CONFIG = {
    "config_list": [
        {"model": QWEN_MODEL, "api_key": Qwen_API, "base_url":  QWEN_URL}
    ],
    "temperature": TEMPERATURE,
    "timeout": TIMEOUT,
}