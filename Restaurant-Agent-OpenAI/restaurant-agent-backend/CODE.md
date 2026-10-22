# Restaurant Assistant Backend

FastAPI backend: an OpenAI tool-calling agent (official `openai` SDK) with restaurant tools. There is no separate frontend; use the Swagger UI page at `/docs` to chat with it. See `../README.md` for setup and usage.

Endpoints: `POST /chat`, `DELETE /chat/{session_id}`, `GET /health`, and `/docs` (Swagger UI). Run: `uvicorn main:app --reload`.

Environment variables (set in `.env` locally or in the host's settings; never commit them): `OPENAI_API_KEY`, and optionally `MODEL`.

## main.py

```python
﻿
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
```

## tools.py

```python
import json

# ============================================================
# TOOL 1: Search Restaurants by Cuisine
# ============================================================

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


# ============================================================
# OpenAI tool definitions + dispatcher
# ============================================================

def _schema(fn, properties: dict, required: list) -> dict:
    return {
        "type": "function",
        "function": {
            "name": fn.__name__,
            "description": " ".join(fn.__doc__.split()),
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required,
            },
        },
    }


TOOL_FUNCTIONS = {
    fn.__name__: fn
    for fn in (
        search_restaurants_by_cuisine,
        get_restaurant_details,
        check_reservation_availability,
    )
}

tools = [
    _schema(
        search_restaurants_by_cuisine,
        {"cuisine_type": {"type": "string"}},
        ["cuisine_type"],
    ),
    _schema(
        get_restaurant_details,
        {"restaurant_name": {"type": "string"}},
        ["restaurant_name"],
    ),
    _schema(
        check_reservation_availability,
        {
            "restaurant_name": {"type": "string"},
            "date": {"type": "string", "description": "YYYY-MM-DD"},
            "party_size": {"type": "integer"},
        },
        ["restaurant_name", "date", "party_size"],
    ),
]


def run_tool(name: str, arguments: str) -> str:
    """Execute a tool by name with JSON-encoded arguments; return a string."""
    fn = TOOL_FUNCTIONS.get(name)
    if fn is None:
        return f"Unknown tool '{name}'."
    try:
        result = fn(**json.loads(arguments or "{}"))
    except Exception as exc:
        return f"Tool '{name}' failed: {exc}"
    return result if isinstance(result, str) else json.dumps(result, ensure_ascii=False)
```

## requirements.txt

```text
fastapi
uvicorn[standard]
openai
pydantic
python-dotenv
```

## .gitignore

```text
.env
__pycache__/
*.pyc
.venv/
graphify-out/
```

