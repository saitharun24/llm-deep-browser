from langchain_core.messages import HumanMessage
import prompts.prompts as prompts

def synthesize_answer(smart_llm, summaries, question):
    if not summaries:
        return ["No information found to answer the question."]

    # Filter out empty summaries
    valid_summaries = [s.strip() for s in summaries if s and s.strip()]
    if not valid_summaries:
        return ["No valid summaries to synthesize."]

    context = "\n\n---\n\n".join(
        f"Source {i+1}:\n{summary}"
        for i, summary in enumerate(valid_summaries)
    )

    try:
        response = smart_llm.invoke([HumanMessage(
            content=prompts.SYNTHESIS_PROMPT.format(
                question=question,
                context=context,
            )
        )])

        return response.content.strip()

    except Exception as e:
        print(f"[WARN] Synthesis failed: {e}")
        return [f"Synthesis error: {e}"]