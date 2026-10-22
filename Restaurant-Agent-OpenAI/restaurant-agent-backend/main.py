
import logging
import os
import uuid

from dotenv import load_dotenv

load_dotenv()

from fastapi import FastAPI, HTTPException
from fastapi.responses import RedirectResponse
from openai import AsyncOpenAI
from pydantic import BaseModel, Field

from tools import run_tool, tools

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

MODEL = os.getenv("MODEL", "gpt-4o-mini")
MAX_TOOL_ROUNDS = 5

client = AsyncOpenAI()  # reads OPENAI_API_KEY from the environment

# Conversation memory, kept on the server so a client such as Swagger UI only
# has to send a session_id instead of re-sending the whole history each time.
# In-memory: it is cleared on restart, and the oldest sessions are dropped.
SESSIONS: dict[str, list[dict]] = {}
MAX_SESSIONS = 200
MAX_HISTORY_MESSAGES = 30

app = FastAPI(
    title="Restaurant Assistant API",
    description=(
        "Chat with the restaurant assistant straight from this page.\n\n"
        "1. Open **POST /chat**, click **Try it out**, type a `message`, "
        "click **Execute**.\n"
        "2. Copy the `session_id` from the response.\n"
        "3. Paste it into `session_id` on your next request to continue the "
        "same conversation.\n"
        "4. Use **DELETE /chat/{session_id}** to start over."
    ),
)


class ChatRequest(BaseModel):
    message: str = Field(examples=["Show me Italian restaurants"])
    session_id: str | None = Field(
        default=None,
        description="Leave empty to start a new chat; reuse the value from "
        "the previous response to continue it.",
    )


class ChatResponse(BaseModel):
    reply: str
    tool_calls: list[dict]
    session_id: str


async def invoke_agent(messages: list[dict]) -> tuple[str, list[dict]]:
    """Run the OpenAI tool-calling loop; return (reply, tool_calls)."""
    convo = [{"role": "system", "content": system_prompt}, *messages]
    tool_calls: list[dict] = []

    for _ in range(MAX_TOOL_ROUNDS):
        response = await client.chat.completions.create(
            model=MODEL,
            messages=convo,
            tools=tools,
        )
        msg = response.choices[0].message

        if not msg.tool_calls:
            return msg.content or "", tool_calls

        convo.append(msg.model_dump(exclude_none=True))
        for call in msg.tool_calls:
            output = run_tool(call.function.name, call.function.arguments)
            tool_calls.append({"name": call.function.name, "output": output})
            convo.append(
                {"role": "tool", "tool_call_id": call.id, "content": output}
            )

    return "Sorry, I could not complete that request.", tool_calls


@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    session_id = req.session_id or uuid.uuid4().hex

    # Known session: continue it. New or unknown id: start a fresh conversation.
    past = SESSIONS.get(session_id, [])
    messages = [*past, {"role": "user", "content": req.message}]

    try:
        reply, tool_calls = await invoke_agent(messages)
    except Exception:
        logger.exception("OpenAI agent invocation failed")
        raise HTTPException(
            status_code=502,
            detail="The restaurant assistant could not process your request.",
        )

    # Save the exchange so the next request with this session_id remembers it.
    messages.append({"role": "assistant", "content": reply})
    SESSIONS.pop(session_id, None)  # re-insert so recent sessions stay newest
    SESSIONS[session_id] = messages[-MAX_HISTORY_MESSAGES:]
    while len(SESSIONS) > MAX_SESSIONS:
        SESSIONS.pop(next(iter(SESSIONS)))

    return ChatResponse(reply=reply, tool_calls=tool_calls, session_id=session_id)


@app.delete("/chat/{session_id}")
async def reset_chat(session_id: str):
    """Forget a conversation so the next message starts fresh."""
    SESSIONS.pop(session_id, None)
    return {"status": "cleared", "session_id": session_id}


@app.get("/", include_in_schema=False)
async def root():
    return RedirectResponse("/docs")


@app.get("/health")
async def health():
    return {"status": "ok"}
