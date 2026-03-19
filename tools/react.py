import re
import json
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from langchain_core.messages import HumanMessage
from pipeline.research_agent import run_web_research
from tools.getstockprice import tool_stock_price
import prompts.prompts as prompts


# ─── Tool Functions ────────────────────────────────────────────────────────────

def tool_current_datetime(input):
    return datetime.now().strftime("%A, %B %d %Y, %I:%M %p")

def tool_web_search(fast_llm, smart_llm, query):
    return run_web_research(fast_llm, smart_llm, query)

def tool_llm_knowledge(smart_llm, query):
    return smart_llm.invoke([HumanMessage(content=query)]).content.strip()


# ─── Tool Registry ─────────────────────────────────────────────────────────────

TOOLS = {
    "current_datetime": {
        "fn": tool_current_datetime,
        "description": "Returns the current date and time. Input can be empty.",
        "requires_input": False,
        "input_type": None,
    },
    "web_search": {
        "fn": tool_web_search,
        "description": "Searches the web and returns summarized results. Use for current events, weather, prices, news, or any product pricing.",
        "requires_input": True,
        "input_type": "search_query",
    },
    "llm_knowledge": {
        "fn": tool_llm_knowledge,
        "description": "Answers using LLM's own knowledge. Use for timeless facts: definitions, history, science, math.",
        "requires_input": True,
        "input_type": "question",
    },
    "stock_price": {
        "fn": tool_stock_price,
        "description": "Fetches live stock price from NSE India. ONLY use for Indian stock market symbols like RELIANCE, TCS, INFY. Do NOT use for product names or hardware.",
        "requires_input": True,
        "input_type": "stock_symbol",
    },
}

# ─── Parser ────────────────────────────────────────────────────────────────────

def parse_react_step(response: str) -> dict:
    response = re.sub(r"<think>.*?</think>", "", response, flags=re.DOTALL).strip()
    result = {}
    thought      = re.search(r"Thought:\s*(.+?)(?=Action:|Final Answer:|$)", response, re.DOTALL)
    action       = re.search(r"Action:\s*(.+?)(?=Action Input:|$)", response, re.DOTALL)
    action_input = re.search(r"Action Input:\s*(.+?)(?=Thought:|Action:|Observation:|Final Answer:|$)", response, re.DOTALL)
    final        = re.search(r"Final Answer:\s*(.+?)$", response, re.DOTALL)
    if thought:      result["thought"]      = thought.group(1).strip()
    if action:       result["action"]       = action.group(1).strip()
    if action_input: result["action_input"] = action_input.group(1).strip()
    if final:        result["final_answer"] = final.group(1).strip()
    return result


# ─── Helpers ───────────────────────────────────────────────────────────────────

def required_tool(smart_llm, question: str) -> str | None:
    """Dynamically determine required tool using smart_llm."""
    try:
        response = smart_llm.invoke([HumanMessage(  # ← smart_llm
            content=prompts.REQUIRED_TOOL_PROMPT.format(
                tools=format_tools_for_prompt(),
                question=question
            )
        )]).content.strip()

        response = re.sub(r"```json|```", "", response).strip()
        parsed = json.loads(response)

        tool = parsed.get("tool")
        reason = parsed.get("reason", "")

        if tool and tool not in TOOLS:
            print(f"[WARN] required_tool returned unknown tool '{tool}', ignoring.")
            return None

        print(f"[Required Tool] {tool} — {reason}")
        return tool

    except Exception as e:
        print(f"[WARN] required_tool failed: {e}, proceeding without forced tool.")
        return None


def validate_tool_input(fast_llm, action: str, action_input: str) -> tuple[bool, str, str | None]:
    """Dynamically validate tool input using fast_llm."""
    tool_meta = TOOLS[action]

    if not tool_meta.get("requires_input"):
        return True, "no input required", None

    if not tool_meta.get("input_type"):
        return True, "no input type defined", None

    if not action_input or not action_input.strip():
        return False, f"{action} requires a non-empty input", "web_search"

    # ← Skip LLM validation for web_search — any non-empty string is valid
    if action == "web_search":
        return True, "search query accepted", None

    try:
        response = fast_llm.invoke([HumanMessage(
            content=prompts.VALIDATE_INPUT_PROMPT.format(
                tool=action,
                description=tool_meta["description"],
                input_type=tool_meta["input_type"],
                input=action_input,
            )
        )]).content.strip()

        response = re.sub(r"```json|```", "", response).strip()
        parsed = json.loads(response)

        is_valid   = parsed.get("valid", True)
        reason     = parsed.get("reason", "")
        suggestion = parsed.get("suggestion")

        if not is_valid:
            print(f"[Validate] ❌ {action}({action_input}) — {reason}")
            if suggestion:
                print(f"[Validate] Suggested tool: {suggestion}")
        else:
            print(f"[Validate] ✅ {action}({action_input}) — {reason}")

        return is_valid, reason, suggestion

    except Exception as e:
        print(f"[WARN] Input validation failed: {e}, proceeding anyway")
        return True, "validation error, proceeding", None


def format_tools_for_prompt() -> str:
    return "\n".join(
        f"- {name}: {meta['description']}"
        for name, meta in TOOLS.items()
    )


def parse_plan(smart_llm, response: str) -> list[dict]:
    response = re.sub(r"<think>.*?</think>", "", response, flags=re.DOTALL).strip()

    def is_valid_plan(plan):
        return (
                isinstance(plan, list) and
                plan and
                all(isinstance(s, dict) for s in plan) and
                all(s.get("tool") in TOOLS for s in plan)  # ← reject unknown tools early
        )

        # ── Strategy 1: Direct JSON parse ─────────────────────────────────────────

    try:
        cleaned = re.sub(r"```json|```", "", response).strip()
        if cleaned.startswith("["):
            plan = json.loads(cleaned)
            if is_valid_plan(plan):
                print("[Plan] Strategy 1 succeeded: direct JSON parse")
                return plan
            elif isinstance(plan, list):
                bad = [s.get("tool") for s in plan if s.get("tool") not in TOOLS]
                print(f"[Plan] Strategy 1 rejected: unknown tools {bad}")
    except Exception:
        pass

        # ── Strategy 2: Extract [...] block from mixed text ────────────────────────
    try:
        match = re.search(r"\[.*\]", response, re.DOTALL)
        if match:
            plan = json.loads(match.group())
            if is_valid_plan(plan):
                print("[Plan] Strategy 2 succeeded: extracted JSON from text")
                return plan
            elif isinstance(plan, list):
                bad = [s.get("tool") for s in plan if s.get("tool") not in TOOLS]
                print(f"[Plan] Strategy 2 rejected: unknown tools {bad}")
    except Exception:
        pass

    # ── Strategy 3: Ask LLM to convert reasoning into JSON ────────────────────
    try:
        print("[Plan] Strategies 1 & 2 failed, asking LLM to convert reasoning to JSON...")
        conversion_prompt = f""""Convert this into a JSON execution plan using ONLY these tools: {list(TOOLS.keys())}

Each step MUST use one of the tools above. No other values are allowed for "tool".

Input:
{response}

Return ONLY a raw JSON array. No explanation. No markdown. No code fences.
First character MUST be [ and last MUST be ]

JSON schema:
[{{"step": 1, "tool": "tool_name", "input": "input string", "depends_on": []}}]

Output:"""

        converted = smart_llm.invoke([HumanMessage(content=conversion_prompt)]).content.strip()
        converted = re.sub(r"<think>.*?</think>", "", converted, flags=re.DOTALL).strip()
        print(f"[Plan] Conversion response:\n{converted}")

        cleaned = re.sub(r"```json|```", "", converted).strip()
        if not cleaned.startswith("["):
            cleaned = "[" + cleaned
        if not cleaned.endswith("]"):
            last = cleaned.rfind("]")
            if last != -1:
                cleaned = cleaned[:last + 1]

        plan = json.loads(cleaned)
        if isinstance(plan, list) and plan:
            for step in plan:
                if "step" not in step or "tool" not in step or "input" not in step:
                    raise ValueError(f"Step missing required fields: {step}")
                if step["tool"] not in TOOLS:
                    raise ValueError(f"Unknown tool: {step['tool']}")
                step.setdefault("depends_on", [])
            print(f"[Plan] Strategy 3 succeeded: converted reasoning to {len(plan)} steps")
            return plan

    except Exception as e:
        print(f"[WARN] Strategy 3 failed: {e}")

    print("[WARN] All parse strategies failed.")
    return []


def parse_plan_with_retry(smart_llm, question: str, max_retries: int = 3) -> list[dict]:
    current_date = tool_current_datetime("")
    for attempt in range(max_retries):
        print(f"[Plan] Attempt {attempt + 1}/{max_retries}")
        try:
            prompt = prompts.PLAN_PROMPT if attempt == 0 else prompts.PLAN_PROMPT_STRICT
            response = smart_llm.invoke([HumanMessage(
                content=prompt.format(
                    tools=format_tools_for_prompt(),
                    question=question,
                    date=current_date
                )
            )]).content.strip()

            print(f"[Plan Raw Response]\n{response}")

            # ── Try to salvage JSON from inside <think> block ──────────────────
            think_match = re.search(r"<think>(.*?)</think>", response, re.DOTALL)
            outside_think = re.sub(r"<think>.*?</think>", "", response, flags=re.DOTALL).strip()

            # Prefer content outside think tags, fall back to inside
            content_to_parse = outside_think if outside_think else (
                think_match.group(1).strip() if think_match else ""
            )

            if not content_to_parse:
                print(f"[WARN] Empty response on attempt {attempt + 1}, retrying...")
                continue

            plan = parse_plan(smart_llm, response)
            if plan: return plan

            print(f"[WARN] Could not parse plan on attempt {attempt + 1}, retrying...")

        except Exception as e:
            print(f"[WARN] Unexpected error on attempt {attempt + 1}: {e}, retrying...")

    print("[WARN] All attempts failed, falling back to iterative ReAct.")
    return []


def resolve_input(input_str: str, results: dict) -> str:
    def replace(match):
        step_num = int(match.group(1))
        return str(results.get(step_num, f"[Step {step_num} result unavailable]"))
    return re.sub(r"\{step(\d+)\}", replace, input_str)


def execute_tool(action: str, action_input: str, fast_llm, smart_llm) -> str:
    if action not in TOOLS:
        return f"Unknown tool '{action}'. Available: {list(TOOLS.keys())}"

    is_valid, reason, suggestion = validate_tool_input(fast_llm, action, action_input)

    if not is_valid:
        if suggestion and suggestion in TOOLS:
            try:
                enriched = fast_llm.invoke([HumanMessage(
                    content=f"Convert this into a good web search query: '{action_input}'\nReturn ONLY the search query, nothing else."
                )]).content.strip()
                enriched_input = enriched if enriched else action_input
                print(f"[Execute] Enriched query: '{action_input}' → '{enriched_input}'")
            except Exception:
                enriched_input = action_input

            print(f"[Execute] Rerouting {action}({action_input}) → {suggestion}({enriched_input})")
            return execute_tool(suggestion, enriched_input, fast_llm, smart_llm)

        return f"Invalid input for {action}: {reason}."

    try:
        fn = TOOLS[action]["fn"]
        if action == "web_search":
            result = fn(fast_llm, smart_llm, action_input)
        elif action == "llm_knowledge":
            result = fn(smart_llm, action_input)
        else:
            result = fn(action_input)

        # ← Temporary debug — find what's returning a list/tuple
        if isinstance(result, (list, tuple)):
            print(f"[DEBUG] {action} returned {type(result)}: {result[:2] if isinstance(result, list) else result}")
            return result[0] if result else ""

        return str(result) if result is not None else ""

    except Exception as e:
        import traceback
        traceback.print_exc()
        return f"Tool error: {e}"


def has_placeholder(text: str) -> bool:
    return bool(re.search(r"\[insert|insert current|\[current|\.\.\.]", text, re.IGNORECASE))


# ─── Parallel ReAct ────────────────────────────────────────────────────────────

def run_parallel_react(fast_llm, smart_llm, question: str) -> str | None:
    print("\n[Thinking] Generating execution plan...")

    plan = parse_plan_with_retry(smart_llm, question)
    if not plan:
        print("[WARN] Could not parse plan, falling back to iterative ReAct.")
        return None

    for step in plan:
        if step.get("tool") not in TOOLS:
            print(f"[WARN] Plan contains unknown tool '{step.get('tool')}', falling back.")
            return None

    # ── Deduplicate ────────────────────────────────────────────────────────────
    seen_tool_inputs = {}
    deduped_plan = []
    step_remap = {}

    for step in plan:
        key = (step["tool"], step["input"])
        if key in seen_tool_inputs:
            canonical = seen_tool_inputs[key]
            step_remap[step["step"]] = canonical
            print(f"[INFO] Deduped step {step['step']} → reusing result of step {canonical}")
        else:
            seen_tool_inputs[key] = step["step"]
            step_remap[step["step"]] = step["step"]
            deduped_plan.append(step)

    for step in deduped_plan:
        step["depends_on"] = list({
            step_remap.get(dep, dep)
            for dep in step.get("depends_on", [])
            if step_remap.get(dep, dep) != step["step"]
        })

    print(f"[Thinking] {len(plan)} steps planned → {len(deduped_plan)} after dedup")

    results = {}
    completed = set()
    pending = list(deduped_plan)

    while pending:
        ready = [
            s for s in pending
            if all(dep in completed for dep in s.get("depends_on", []))
        ]

        if not ready:
            print("[WARN] Deadlock in plan, falling back to iterative ReAct.")
            return None

        print(f"\n[Thinking] Running {len(ready)} steps in parallel: {[s['step'] for s in ready]}")

        with ThreadPoolExecutor(max_workers=len(ready)) as executor:
            futures = {
                executor.submit(
                    execute_tool,
                    s["tool"],
                    resolve_input(s["input"], results),
                    fast_llm,
                    smart_llm,
                ): s for s in ready
            }
            for future in as_completed(futures):
                step = futures[future]
                step_num = step["step"]
                try:
                    observation = future.result()
                    results[step_num] = observation
                    completed.add(step_num)
                    print(f"[Step {step_num}] ✅ {step['tool']}({step['input'][:50]})")
                    print(f"[Observation] {str(observation)[:200]}...")
                except Exception as e:
                    results[step_num] = f"[ERROR] {e}"
                    completed.add(step_num)
                    print(f"[Step {step_num}] ❌ Failed: {e}")

        pending = [s for s in pending if s["step"] not in completed]

    # ── Synthesize ─────────────────────────────────────────────────────────────
    print("\n[Thinking] Synthesizing final answer...")
    observations_text = "\n\n".join(
        f"Step {num} ({deduped_plan[i]['tool']}): {obs}"
        for i, (num, obs) in enumerate(sorted(results.items()))
    )

    final = smart_llm.invoke([HumanMessage(
        content=prompts.SYNTHESIS_PROMPT.format(
            question=question,
            context=observations_text
        )
    )]).content.strip()

    final = re.sub(r"<think>.*?</think>", "", final, flags=re.DOTALL).strip()

    if has_placeholder(final):
        print("[WARN] Synthesis returned placeholder, falling back to iterative ReAct.")
        return None

    return final


# ─── Iterative ReAct ──────────────────────────────────────────────────────────

def run_iterative_react(fast_llm, smart_llm, question: str, max_steps: int = 10) -> str:
    print("\n[Thinking] Starting loop...")
    history = ""
    observations_made = 0
    forced_tool = required_tool(smart_llm, question)
    forced_tool_satisfied = False
    used_tool_inputs = {}
    current_date = tool_current_datetime("")
    parse_failures = 0
    MAX_PARSE_FAILURES = 3

    for step in range(max_steps):
        print(f"\n[Thinking] Step {step + 1}")

        prompt = prompts.REACT_PROMPT.format(
            tools=format_tools_for_prompt(),
            question=question,
            history=history if history else "",
            date=current_date
        )

        response = smart_llm.invoke([HumanMessage(content=prompt)]).content.strip()
        print(f"[LLM Response]\n{response}")

        parsed = parse_react_step(response)

        # ── Final Answer ───────────────────────────────────────────────────────
        if "final_answer" in parsed:
            answer = parsed["final_answer"]

            if forced_tool and not forced_tool_satisfied:
                print(f"[WARN] '{forced_tool}' is required but not used, forcing...")
                history += f"\nSystem: You MUST use '{forced_tool}' tool before answering. Do not answer from memory.\n"
                continue

            if has_placeholder(answer):
                print("[WARN] Placeholder answer detected, forcing retry...")
                history += "\nSystem: Your Final Answer contained placeholders. Use ONLY data from Observations.\n"
                continue

            # Always synthesize final answer with smart_llm
            print(f"\n[Thinking] Synthesizing final answer with smart_llm...")
            final = smart_llm.invoke([HumanMessage(
                content=prompts.SYNTHESIS_PROMPT.format(
                    question=question,
                    context=history
                )
            )]).content.strip()

            final = re.sub(r"<think>.*?</think>", "", final, flags=re.DOTALL).strip()
            print(f"\n[Thinking] ✅ Final Answer: {final}")
            return final

        # ── Unparseable response ───────────────────────────────────────────────
        if "action" not in parsed and "final_answer" not in parsed:
            parse_failures += 1
            print(f"[WARN] Could not parse response ({parse_failures}/{MAX_PARSE_FAILURES}), retrying...")

            if parse_failures >= MAX_PARSE_FAILURES:
                print("[WARN] Too many parse failures, stopping.")
                break

            history += "\nSystem: Your last response could not be parsed. You MUST respond in exactly this format:\nThought: <reasoning>\nAction: <tool_name>\nAction Input: <input>\n"
            continue

        # ── Reset parse failure counter on success ─────────────────────────────
        parse_failures = 0

        # ── Tool Call ──────────────────────────────────────────────────────────
        if "action" not in parsed:
            print("[WARN] Could not parse action, stopping.")
            break

        action       = parsed["action"].lower().strip()
        action_input = parsed.get("action_input", "").strip()
        thought      = parsed.get("thought", "")

        if action in used_tool_inputs and action_input in used_tool_inputs[action]:
            print(f"[WARN] Detected repeat call: {action}({action_input}), forcing Final Answer...")
            history += f"\nSystem: You already called '{action}' with input '{action_input}' and received an observation. Do NOT call it again. Give the Final Answer now.\n"
            continue

        used_tool_inputs.setdefault(action, set()).add(action_input)

        observation = execute_tool(action, action_input, fast_llm, smart_llm)
        observations_made += 1

        if forced_tool and action == forced_tool:
            forced_tool_satisfied = True

        MAX_OBS_LENGTH = 2000

        obs_str = str(observation)
        if len(obs_str) > MAX_OBS_LENGTH:
            obs_str = obs_str[:MAX_OBS_LENGTH] + "\n...[truncated]"

        print(f"[Tool] {action}({action_input})")
        print(f"[Observation] {str(observation)[:200]}...")

        history += f"\nThought: {thought}\nAction: {action}\nAction Input: {action_input}\nObservation: {obs_str}\n"

    return "Could not determine an answer after maximum steps."

# ─── Unified Entry Point ───────────────────────────────────────────────────────

def run_research(fast_llm, smart_llm, question: str) -> str:
    answer = run_parallel_react(fast_llm, smart_llm, question)
    if answer is None:
        answer = run_iterative_react(fast_llm, smart_llm, question)
    return answer