from langchain_openai import ChatOpenAI

def get_groq_llm():
    """Create a Groq LLM instance optimized for RAG integration."""
    return ChatOpenAI(
        temperature=0,
        model="qwen/qwen3-32b",
        api_key="your api key here",
        base_url="https://api.groq.com/openai/v1",
        max_completion_tokens=2048
    )
