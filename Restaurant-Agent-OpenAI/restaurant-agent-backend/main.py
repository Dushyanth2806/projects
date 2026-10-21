
import logging
import os

from dotenv import load_dotenv

load_dotenv()

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
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

app = FastAPI(title="Restaurant Assistant API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("CORS_ORIGINS", "*").split(","),
    allow_methods=["*"],
    allow_headers=["*"],
)


class Message(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    message: str
    history: list[Message] = Field(default_factory=list)


class ChatResponse(BaseModel):
    reply: str
    tool_calls: list[dict]


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
    messages = [m.model_dump() for m in req.history]
    messages.append({"role": "user", "content": req.message})

    try:
        reply, tool_calls = await invoke_agent(messages)
    except Exception:
        logger.exception("OpenAI agent invocation failed")
        raise HTTPException(
            status_code=502,
            detail="The restaurant assistant could not process your request.",
        )

    return ChatResponse(reply=reply, tool_calls=tool_calls)


@app.get("/health")
async def health():
    return {"status": "ok"}
