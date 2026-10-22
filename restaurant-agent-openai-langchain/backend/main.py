
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
    os.getenv("MODEL", "openai:gpt-4o-mini")
)

agent = create_agent(
    model=model,
    tools=tools,
    system_prompt=system_prompt,
)

MAX_SESSIONS = 200
MAX_HISTORY_MESSAGES = 30

# session_id -> list of {"role", "content"} messages (in memory, lost on restart)
SESSIONS: dict[str, list[dict]] = {}

API_DESCRIPTION = """
Chat with a restaurant recommendation assistant, right from this page.

1. Open **POST /chat** -> **Try it out** -> type a message -> **Execute**.
2. Copy the `session_id` from the response.
3. Paste it into `session_id` on your next request to continue the conversation.
4. Call **DELETE /chat/{session_id}** to start over.
"""

app = FastAPI(title="Restaurant Assistant API", description=API_DESCRIPTION)


class ChatRequest(BaseModel):
    message: str = Field(examples=["Show me Italian restaurants"])
    session_id: str | None = Field(
        default=None,
        description=(
            "Leave empty to start a new conversation. To continue one, paste "
            "the session_id returned by a previous response."
        ),
    )


class ChatResponse(BaseModel):
    reply: str
    tool_calls: list[dict]
    session_id: str


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


async def invoke_agent(messages: list[dict]):
    return await agent.ainvoke({"messages": messages})


@app.get("/", include_in_schema=False)
async def root():
    return RedirectResponse(url="/docs")


@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    session_id = req.session_id or uuid.uuid4().hex
    messages = list(SESSIONS.get(session_id, []))
    messages.append({"role": "user", "content": req.message})

    try:
        result = await invoke_agent(messages)
    except Exception:
        logger.exception("OpenAI agent invocation failed")
        raise HTTPException(
            status_code=502,
            detail="The restaurant assistant could not process your request.",
        )

    out = result["messages"]

    tool_calls = [
        {
            "name": m.name,
            "output": _text(m.content),
        }
        for m in out
        if getattr(m, "type", None) == "tool"
    ]

    reply = _text(out[-1].content) if out else ""

    messages.append({"role": "assistant", "content": reply})
    SESSIONS.pop(session_id, None)
    SESSIONS[session_id] = messages[-MAX_HISTORY_MESSAGES:]
    while len(SESSIONS) > MAX_SESSIONS:
        del SESSIONS[next(iter(SESSIONS))]

    return ChatResponse(
        reply=reply,
        tool_calls=tool_calls,
        session_id=session_id,
    )


@app.delete("/chat/{session_id}")
async def clear_chat(session_id: str):
    SESSIONS.pop(session_id, None)
    return {"status": "cleared", "session_id": session_id}


@app.get("/health")
async def health():
    return {"status": "ok"}
