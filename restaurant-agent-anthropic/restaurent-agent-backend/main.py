
import json
import logging
import os
import uuid

from dotenv import load_dotenv

load_dotenv()

from fastapi import FastAPI, HTTPException
from fastapi.responses import RedirectResponse
from anthropic import AsyncAnthropic
from pydantic import BaseModel, Field

from tools import tool_definitions, tool_functions

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

system_prompt = """
You are a helpful restaurant recommendation assistant.

You have access to these tools:

1. Search restaurants by cuisine
2. Get restaurant details
3. Check reservation availability

Use the appropriate tool whenever necessary.

Do not invent restaurant information.

If the available tools cannot answer the question,
clearly say that the requested information is outside
your available capabilities.

Be friendly, conversational, and concise.
"""

MODEL = os.getenv("MODEL", "claude-haiku-5-5")
MAX_TOOL_ROUNDS = 5

client = AsyncAnthropic()  # reads ANTHROPIC_API_KEY from the environment

MAX_SESSIONS = 200
MAX_HISTORY_MESSAGES = 30

# session_id -> plain-text turns only: [{"role": "user"|"assistant", "content": str}, ...]
SESSIONS: dict[str, list[dict]] = {}

API_DESCRIPTION = """
A restaurant recommendation assistant powered by Claude. Use this page to chat with it.

**How to use**

1. `POST /chat` -> **Try it out** -> type a message -> **Execute**.
2. Copy the `session_id` from the response.
3. Paste it into `session_id` on your next request to continue the conversation.
4. `DELETE /chat/{session_id}` to clear the conversation and start over.
"""

app = FastAPI(title="Restaurant Assistant API", description=API_DESCRIPTION)


class ChatRequest(BaseModel):
    message: str = Field(examples=["Show me Italian restaurants"])
    session_id: str | None = Field(
        default=None,
        description=(
            "Leave empty to start a new conversation. To continue one, paste the "
            "session_id returned by a previous response."
        ),
    )


class ChatResponse(BaseModel):
    reply: str
    tool_calls: list[dict]
    session_id: str


def run_tool(name: str, args: dict) -> str:
    fn = tool_functions.get(name)
    if fn is None:
        return f"Unknown tool: {name}"
    try:
        result = fn(**args)
    except Exception as e:
        return f"Tool error: {e}"
    return result if isinstance(result, str) else json.dumps(result, ensure_ascii=False)


async def invoke_claude_agent(messages: list[dict]) -> tuple[str, list[dict]]:
    """Run Claude with tools until it gives a final answer. Returns (reply, tool_calls)."""
    tool_calls: list[dict] = []

    for _ in range(MAX_TOOL_ROUNDS):
        response = await client.messages.create(
            model=MODEL,
            max_tokens=1024,
            system=system_prompt,
            tools=tool_definitions,
            messages=messages,
        )

        if response.stop_reason != "tool_use":
            reply = "".join(b.text for b in response.content if b.type == "text")
            return reply, tool_calls

        messages.append({"role": "assistant", "content": response.content})

        results = []
        for block in response.content:
            if block.type == "tool_use":
                output = run_tool(block.name, block.input)
                tool_calls.append({"name": block.name, "output": output})
                results.append(
                    {"type": "tool_result", "tool_use_id": block.id, "content": output}
                )
        messages.append({"role": "user", "content": results})

    return "Sorry, I couldn't complete that request.", tool_calls


def save_turn(session_id: str, history: list[dict], user_text: str, reply: str) -> None:
    """Store the plain-text turn, keep the newest sessions, and cap the history."""
    history = history + [
        {"role": "user", "content": user_text},
        {"role": "assistant", "content": reply},
    ]
    history = history[-MAX_HISTORY_MESSAGES:]
    # The API needs the conversation to start with a user message.
    while history and history[0]["role"] != "user":
        history = history[1:]

    SESSIONS.pop(session_id, None)  # re-insert so recent sessions stay newest
    SESSIONS[session_id] = history
    while len(SESSIONS) > MAX_SESSIONS:
        SESSIONS.pop(next(iter(SESSIONS)))


@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    session_id = req.session_id or uuid.uuid4().hex
    history = SESSIONS.get(session_id, [])
    # Work on a copy: the agent loop appends tool_use/tool_result blocks to it.
    messages = [*history, {"role": "user", "content": req.message}]

    try:
        reply, tool_calls = await invoke_claude_agent(messages)
    except Exception:
        logger.exception("Claude agent invocation failed")
        raise HTTPException(
            status_code=502,
            detail="The restaurant assistant could not process your request.",
        )

    if reply.strip():  # an empty assistant message would be invalid for the API
        save_turn(session_id, history, req.message, reply)

    return ChatResponse(reply=reply, tool_calls=tool_calls, session_id=session_id)


@app.delete("/chat/{session_id}")
async def clear_chat(session_id: str):
    """Clear a conversation so the next message starts fresh."""
    cleared = SESSIONS.pop(session_id, None) is not None
    return {"session_id": session_id, "cleared": cleared}


@app.get("/", include_in_schema=False)
async def root():
    return RedirectResponse(url="/docs")


@app.get("/health")
async def health():
    return {"status": "ok"}
