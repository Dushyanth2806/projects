# Restaurant Assistant Backend

FastAPI backend: a Claude agent (LangChain + `langchain-anthropic`) with restaurant tools. You chat with it through the Swagger UI at `/docs`. Conversation memory is kept on the server, so you never paste earlier messages.

## Endpoints

| Method | Path | Description |
|---|---|---|
| GET | `/` | Redirects to `/docs` |
| POST | `/chat` | Send `message` (and optional `session_id`). Returns `reply`, `tool_calls`, `session_id` |
| DELETE | `/chat/{session_id}` | Clear a conversation |
| GET | `/health` | Health check |

Run: `uvicorn main:app --reload --port 8000` then open http://localhost:8000/docs.

## Environment variables

Set in `.env` locally (never commit it): `ANTHROPIC_API_KEY` (required). Optional: `MODEL` (default `anthropic:claude-haiku-5-5`).
## main.py

```python

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
```

## tools.py

```python
from langchain_core.tools import tool

# ============================================================
# TOOL 1: Search Restaurants by Cuisine
# ============================================================

@tool
def search_restaurants_by_cuisine(cuisine_type: str) -> list:
    """
    Search for restaurants by cuisine type.
    Supported cuisines: Indian, Italian, Chinese, Mexican.
    """

    restaurants = {
        "indian": [
            {"name": "Spice Palace", "rating": 4.5, "price": "₹₹"},
            {"name": "Curry House", "rating": 4.3, "price": "₹₹"},
            {"name": "Taj Flavors", "rating": 4.7, "price": "₹₹₹"}
        ],
        "italian": [
            {"name": "Pizza Bella", "rating": 4.4, "price": "₹₹"},
            {"name": "Pasta Dreams", "rating": 4.6, "price": "₹₹₹"},
            {"name": "Roma Kitchen", "rating": 4.2, "price": "₹₹"}
        ],
        "chinese": [
            {"name": "Dragon House", "rating": 4.5, "price": "₹₹"},
            {"name": "Ming's Kitchen", "rating": 4.3, "price": "₹₹"},
            {"name": "Golden Wok", "rating": 4.8, "price": "₹₹₹"}
        ],
        "mexican": [
            {"name": "Taco Fiesta", "rating": 4.4, "price": "₹₹"},
            {"name": "Salsa Nights", "rating": 4.6, "price": "₹₹₹"},
            {"name": "Burrito House", "rating": 4.2, "price": "₹₹"}
        ]
    }

    cuisine_lower = cuisine_type.lower().strip()

    return restaurants.get(cuisine_lower, [])


# ============================================================
# TOOL 2: Get Restaurant Details
# ============================================================

@tool
def get_restaurant_details(restaurant_name: str) -> dict:
    """
    Get detailed information about a restaurant,
    including address, phone, hours and specialties.
    """

    details = {
        "Spice Palace": {
            "cuisine": "Indian",
            "rating": 4.5,
            "address": "123 Food Street, City Center",
            "phone": "9876543210",
            "hours": "11 AM - 11 PM",
            "speciality": "Butter Chicken, Biryani"
        },

        "Pizza Bella": {
            "cuisine": "Italian",
            "rating": 4.4,
            "address": "456 Dine Avenue, Downtown",
            "phone": "9123456789",
            "hours": "12 PM - 10:30 PM",
            "speciality": "Margherita Pizza, Tiramisu"
        },

        "Dragon House": {
            "cuisine": "Chinese",
            "rating": 4.5,
            "address": "789 Noodle Lane, Market Area",
            "phone": "9111222333",
            "hours": "11:30 AM - 11 PM",
            "speciality": "Hakka Noodles, Fried Rice"
        },

        "Taco Fiesta": {
            "cuisine": "Mexican",
            "rating": 4.4,
            "address": "321 Spice Row, Hip District",
            "phone": "9222333444",
            "hours": "1 PM - 12 AM",
            "speciality": "Fish Tacos, Guacamole"
        },

        "Curry House": {
            "cuisine": "Indian",
            "rating": 4.3,
            "address": "555 Flavor Street, Old Town",
            "phone": "9333444555",
            "hours": "11 AM - 10 PM",
            "speciality": "Paneer Tikka, Dal Makhani"
        },

        "Taj Flavors": {
            "cuisine": "Indian",
            "rating": 4.7,
            "address": "999 Taste Lane, Premium Zone",
            "phone": "9444555666",
            "hours": "12 PM - 11 PM",
            "speciality": "Biryani, Tandoori Chicken"
        },

        "Golden Wok": {
            "cuisine": "Chinese",
            "rating": 4.8,
            "address": "777 Wok Street, Central Market",
            "phone": "9555666777",
            "hours": "11 AM - 11 PM",
            "speciality": "Szechuan Noodles, Dim Sum"
        }
    }

    return details.get(
        restaurant_name.strip(),
        {"error": f"Restaurant '{restaurant_name}' not found"}
    )


# ============================================================
# TOOL 3: Check Reservation Availability
# ============================================================

@tool
def check_reservation_availability(
    restaurant_name: str,
    date: str,
    party_size: int
) -> str:
    """
    Check if a restaurant has reservation availability
    for a specific date and number of people.
    """

    available_slots = {
        "Spice Palace": {
            "2024-11-15": True,
            "2024-11-16": False,
            "2024-11-17": True
        },

        "Pizza Bella": {
            "2024-11-15": False,
            "2024-11-16": True,
            "2024-11-17": True
        },

        "Dragon House": {
            "2024-11-15": True,
            "2024-11-16": True,
            "2024-11-17": False
        },

        "Taco Fiesta": {
            "2024-11-15": True,
            "2024-11-16": True,
            "2024-11-17": True
        },

        "Curry House": {
            "2024-11-15": True,
            "2024-11-16": False,
            "2024-11-17": True
        },

        "Taj Flavors": {
            "2024-11-15": False,
            "2024-11-16": True,
            "2024-11-17": True
        }
    }

    if restaurant_name not in available_slots:
        return f"Restaurant '{restaurant_name}' not found."

    is_available = available_slots[restaurant_name].get(
        date,
        False
    )

    if is_available:
        return (
            f"✓ {restaurant_name} has availability "
            f"for {party_size} people on {date}"
        )

    return (
        f"✗ {restaurant_name} is fully booked "
        f"for {date}"
    )


tools = [search_restaurants_by_cuisine, get_restaurant_details, check_reservation_availability]
```

## requirements.txt

```text
fastapi
uvicorn[standard]
langchain>=1.0
langchain-anthropic
pydantic
python-dotenv
```

## .env.example

```text
ANTHROPIC_API_KEY=your-api-key-here
# Optional. Defaults to anthropic:claude-haiku-5-5
MODEL=anthropic:claude-haiku-5-5
```
