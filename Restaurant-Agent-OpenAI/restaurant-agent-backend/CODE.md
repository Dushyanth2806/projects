# Restaurant Assistant Backend

FastAPI backend: an OpenAI tool-calling agent (official `openai` SDK) with restaurant tools.

Endpoints: `POST /chat` and `GET /health`. Run: `uvicorn main:app --reload`.

Environment variables (set in `.env` locally or in the host's settings; never commit them): `OPENAI_API_KEY`, `CORS_ORIGINS`.

## main.py

```python
﻿
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

