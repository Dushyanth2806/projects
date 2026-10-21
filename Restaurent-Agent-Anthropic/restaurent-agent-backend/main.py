
import json
import logging
import os

from dotenv import load_dotenv

load_dotenv()

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
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


@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    messages = [m.model_dump() for m in req.history]
    messages.append({"role": "user", "content": req.message})

    try:
        reply, tool_calls = await invoke_claude_agent(messages)
    except Exception:
        logger.exception("Claude agent invocation failed")
        raise HTTPException(
            status_code=502,
            detail="The restaurant assistant could not process your request.",
        )

    return ChatResponse(reply=reply, tool_calls=tool_calls)


@app.get("/health")
async def health():
    return {"status": "ok"}
