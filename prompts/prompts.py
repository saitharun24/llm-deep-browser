# ─── Prompts ───────────────────────────────────────────────────────────────────

PLAN_PROMPT = """Today is {date}. You are a task planner. Analyze this question and create an execution plan.

Available tools:
{tools}

TOOL USAGE PRIORITY (follow this strictly in order):
1. current_datetime — ALWAYS use for any date or time question, never guess
2. stock_price      — ALWAYS use for any stock, share price, or market data
3. web_search       — ALWAYS use for weather, news, prices, current events, or anything that changes over time
4. llm_knowledge    — ONLY use for timeless facts: math, definitions, history, science concepts
5. No tool          — ONLY if the question is purely conversational e.g. "how are you"

WHEN IN DOUBT — use a tool. It is always better to verify than to guess.

For each piece of information needed, decide:
1. Which tool to use
2. What input to pass
3. Whether it depends on another step's output (if so, which step number)

Return ONLY a valid JSON array like this:
[
  {{"step": 1, "tool": "current_datetime", "input": "", "depends_on": []}},
  {{"step": 2, "tool": "web_search", "input": "Chennai weather", "depends_on": []}},
  {{"step": 3, "tool": "llm_knowledge", "input": "Summarize: {{step1}} {{step2}}", "depends_on": [1, 2]}}
]

Rules:
- Steps with no dependencies can run in parallel
- Steps that need output from other steps must list them in depends_on
- Use {{stepN}} as placeholder in input when you need output from step N
- For single-tool questions, return a single step plan
- NEVER include a step with tool "none" or "direct" — use llm_knowledge instead
- NEVER skip tools for real-time data like dates, weather, stocks

Question: {question}"""


REACT_PROMPT = """Today is {date}. You are a research assistant. Answer the question using the ReAct framework.

Available tools:
{tools}

TOOL USAGE PRIORITY (follow this strictly in order):
1. current_datetime — ALWAYS use for any date or time question, never guess
2. stock_price      — ALWAYS use for any stock, share price, or market data
3. web_search       — ALWAYS use for weather, news, prices, current events, or anything that changes over time
4. llm_knowledge    — ONLY use for timeless facts: math, definitions, history, science concepts
5. Direct Answer    — ONLY if the question is purely conversational e.g. "how are you"

WHEN IN DOUBT — use a tool. It is always better to verify than to guess.

Format each step EXACTLY like this:
Thought: <your reasoning, explicitly state WHY you chose this tool over direct answer>
Action: <tool_name>
Action Input: <input to the tool>

When you have enough information, respond with:
Thought: I now have enough information to answer.
Final Answer: <your complete answer using ONLY data from Observations>

STRICT RULES:
- Output ONE Thought + ONE Action + ONE Action Input per response, then STOP
- Do NOT output an Observation yourself — wait for it to be provided
- If you see past Observations below, they are REAL data you already received
- NEVER say you haven't received results if an Observation is present below
- NEVER answer from memory for: dates, times, prices, weather, news, stocks
- NEVER fill in placeholders like [insert ...] or [current ...]
- Final Answer must only contain facts from Observations, not from your training data

Question: {question}
Previous Steps: {history}

Continue:"""

REQUIRED_TOOL_PROMPT = """You are a tool selection assistant.

Available tools and their descriptions:
{tools}

Given a question, identify which single tool is MOST appropriate to answer it.
Consider:
- If the question involves real-time or current data, prefer specific tools over web_search
- If a dedicated tool exists for the data type (e.g. stock_price for stocks), always prefer it over web_search
- Only return web_search if no more specific tool exists
- Return null if no tool is needed (purely conversational)

Return ONLY a JSON object like this:
{{"tool": "tool_name", "reason": "one line reason"}}
or
{{"tool": null, "reason": "one line reason"}}

Question: {question}"""


VALIDATE_INPUT_PROMPT = """You are an input validator for a tool.

Tool: {tool}
Tool description: {description}
Input type expected: {input_type}
Provided input: "{input}"

Is this input appropriate for this tool?
Answer ONLY with a JSON object:
{{"valid": true/false, "reason": "one line reason", "suggestion": "better tool name if invalid, else null"}}

Output:"""

PLAN_PROMPT_STRICT = """Today is {date}. Output a JSON array for this question. Start with [ immediately.
Tools: {tools}
Question: {question}
Each step must have: step, tool, input, depends_on
["""

SYNTHESIS_PROMPT = """You are a research assistant. Using ONLY the research provided below, answer the question thoroughly.

QUESTION:
{question}

RESEARCH:
{context}

INSTRUCTIONS:
- Answer directly and comprehensively
- Include all relevant facts, numbers, and data from the research
- Structure your answer with clear sections if the answer is long
- If the research contains conflicting information, mention both perspectives
- Do NOT add information from your own training data
- Do NOT include placeholders or say "based on the research"
- Just answer the question directly using the facts provided

ANSWER:"""

SUMMARIZE_PROMPT = """Extract and summarize all important information from the content below.

PRESERVE:
- All numbers, prices, dates, percentages, statistics
- Product names, model numbers, specifications
- Company names, people, locations
- Key facts and findings
- Any lists or comparisons

SKIP:
- Navigation menus, ads, cookie notices
- Repetitive boilerplate text
- Social media buttons, share prompts

CONTENT:
{text}

SUMMARY:"""