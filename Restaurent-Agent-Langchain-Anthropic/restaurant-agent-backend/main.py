
import logging
import os
import uuid

from dotenv import load_dotenv

load_dotenv()

from fastapi import FastAPI, HTTPException
from fastapi.responses import RedirectResponse
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from pydantic import BaseModel, Field

from tools import tools

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

model = init_chat_model(
    os.getenv("MODEL", "anthropic:claude-haiku-5-5")
)

agent = create_agent(
    model=model,
    tools=tools,
    system_prompt=system_prompt,
)

# ------------------------------------------------------------
# Conversation memory (in RAM, lost on restart)
# session_id -> [{"role": "user"|"assistant", "content": "<text>"}, ...]
# Only plain-text turns are stored, always as user/assistant pairs, so a
# saved session is always a valid message sequence for the Anthropic API.
# ------------------------------------------------------------
MAX_SESSIONS = 200
MAX_HISTORY_MESSAGES = 30

SESSIONS: dict[str, list[dict]] = {}

description = """
Chat with a restaurant assistant (LangChain + Claude) right from this page.

**How to use it**

1. `POST /chat` -> **Try it out** -> type a message -> **Execute**.
2. Copy the `session_id` from the response.
3. Paste it into `session_id` on your next request to continue the conversation.
4. `DELETE /chat/{session_id}` to start over.
"""

app = FastAPI(title="Restaurant Assistant API", description=description)


class ChatRequest(BaseModel):
    message: str = Field(
        min_length=1,
        description="Your message to the assistant.",
        examples=["Show me Italian restaurants"],
    )
    session_id: str | None = Field(
        default=None,
        max_length=64,
        description=(
            "Leave empty to start a new conversation. To continue one, paste "
            "the session_id returned by a previous response."
        ),
    )


class ChatResponse(BaseModel):
    reply: str
    tool_calls: list[dict]
    session_id: str = Field(
        description="Send this back as session_id to continue the conversation."
    )


def _text(content) -> str:
    if isinstance(content, str):
        return content

    if isinstance(content, list):
        return "".join(
            block.get("text", "")
            for block in content
            if isinstance(block, dict)
        )

    return str(content) if content is not None else ""


async def invoke_claude_agent(messages: list[dict]):
    return await agent.ainvoke({"messages": messages})


def _save_turn(session_id: str, past: list[dict], user_text: str, reply: str) -> None:
    turns = past + [
        {"role": "user", "content": user_text},
        {"role": "assistant", "content": reply},
    ]

    # Trim whole user/assistant pairs from the front so the conversation
    # always starts with a "user" message and keeps alternating.
    excess = len(turns) - MAX_HISTORY_MESSAGES
    if excess > 0:
        turns = turns[excess + (excess % 2):]

    # Re-insert the key so the most recently used sessions stay newest.
    SESSIONS.pop(session_id, None)
    SESSIONS[session_id] = turns

    while len(SESSIONS) > MAX_SESSIONS:
        del SESSIONS[next(iter(SESSIONS))]


@app.get("/", include_in_schema=False)
async def root():
    return RedirectResponse(url="/docs")


@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    session_id = req.session_id or uuid.uuid4().hex

    past = SESSIONS.get(session_id, [])
    messages = past + [{"role": "user", "content": req.message}]

    try:
        result = await invoke_claude_agent(messages)
    except Exception:
        logger.exception("Claude agent invocation failed")
        raise HTTPException(
            status_code=502,
            detail="The restaurant assistant could not process your request.",
        )

    out = result["messages"]

    # Tool calls made while answering this message only (not earlier turns).
    tool_calls = [
        {
            "name": m.name,
            "output": _text(m.content),
        }
        for m in out[len(messages):]
        if getattr(m, "type", None) == "tool"
    ]

    reply = _text(out[-1].content) if out else ""

    # An empty assistant message is invalid for the Anthropic API,
    # so don't save a turn that has no reply text.
    if reply.strip():
        _save_turn(session_id, past, req.message, reply)

    return ChatResponse(
        reply=reply,
        tool_calls=tool_calls,
        session_id=session_id,
    )


@app.delete("/chat/{session_id}")
async def clear_chat(session_id: str):
    cleared = SESSIONS.pop(session_id, None) is not None
    return {"session_id": session_id, "cleared": cleared}


@app.get("/health")
async def health():
    return {"status": "ok"}
