import re
from langchain_core.messages import HumanMessage
import prompts.prompts as prompts

def summarize_chunk(fast_llm, text: str) -> str:
    if not text or not text.strip():
        return ""
    print(text)
    try:
        response = fast_llm.invoke([HumanMessage(
            content=prompts.SUMMARIZE_PROMPT.format(text=text)
        )])

        return response.content.strip()

    except Exception as e:
        print(f"[WARN] Summarization failed: {e}")
        return ""