# Restaurant Assistant (LangChain + Claude)

A chat assistant that recommends restaurants, shows their details and checks reservation availability. It is a FastAPI backend with a LangChain agent powered by an Anthropic Claude model. There is no separate frontend: you chat with it through FastAPI's built-in **Swagger UI** at `/docs`.

## How it works

```
 You (Swagger UI /docs)
        |
        |  POST /chat  { message, session_id? }
        v
 FastAPI  (main.py)  <---- session memory (in RAM, per session_id)
        |
        |  past messages + new message
        v
 LangChain agent  (create_agent)
        |            \
        |             +--> tools.py: search_restaurants_by_cuisine
        |             +--> tools.py: get_restaurant_details
        |             +--> tools.py: check_reservation_availability
        v
 Anthropic Claude model  (langchain-anthropic)
        |
        v
 { reply, tool_calls, session_id }
```

1. You send a message. If you include a `session_id`, the server loads that conversation's earlier messages.
2. The LangChain agent sends the conversation to Claude. Claude decides whether to call a tool.
3. LangChain runs the tool, gives the result back to Claude, and Claude writes the final reply.
4. The server saves your message and the final reply (plain text only) under the `session_id`, and returns the reply, the tools used and the `session_id`.

## Project structure

```
.
├── README.md
└── restaurant-agent-backend/
    ├── main.py            FastAPI app, LangChain agent, session memory
    ├── tools.py           The three restaurant tools (sample data)
    ├── requirements.txt   Python dependencies
    ├── .env.example       Template for your .env (placeholders only)
    ├── .env               Your real settings (not committed)
    └── CODE.md            All source files in one document
```

## Setup

Requires Python 3.10 or newer.

```bash
cd restaurant-agent-backend

# 1. Create and activate a virtual environment
python -m venv .venv
# Windows (PowerShell):  .venv\Scripts\Activate.ps1
# macOS / Linux:         source .venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Create your .env from the template, then put your key in it
cp .env.example .env          # Windows: copy .env.example .env
```

## Environment variables

| Variable | Required | Description |
|---|---|---|
| `ANTHROPIC_API_KEY` | Yes | Your Anthropic API key. Never commit it. |
| `MODEL` | No | Model to use, in LangChain format. Default: `anthropic:claude-haiku-5-5` |

## Run

```bash
uvicorn main:app --reload --port 8000
```

Open http://localhost:8000/docs (the root URL `/` redirects there).

## Using Swagger UI as the frontend

1. Open http://localhost:8000/docs.
2. Click **POST /chat**, then **Try it out**.
3. In the request body, type your message, for example `"Show me Italian restaurants"`. Leave `session_id` empty (or remove that line) for a new conversation.
4. Click **Execute**. The reply appears under **Response body**.
5. Copy the `session_id` from the response.
6. For your next message, paste that value into `session_id` in the request body and execute again. The assistant remembers the earlier messages, so you can ask "What is the phone number of the highest rated one?".
7. To start over, open **DELETE /chat/{session_id}**, click **Try it out**, paste the `session_id` and **Execute**.

## API

| Method | Path | Description |
|---|---|---|
| GET | `/` | Redirects to `/docs` |
| POST | `/chat` | Body: `message` (required), `session_id` (optional). Returns `reply`, `tool_calls`, `session_id` |
| DELETE | `/chat/{session_id}` | Clears that conversation. Returns `{"session_id": ..., "cleared": true/false}` |
| GET | `/health` | Returns `{"status": "ok"}` |

`tool_calls` lists the tools used while answering that message, each with its `name` and `output`.

## Notes and limits

- **Memory lives in RAM.** It is lost when the server restarts. At most 200 sessions are kept (the oldest are dropped) and each keeps the latest 30 messages. Only the plain text of your messages and the replies is stored, not the tool calls.
- **No authentication.** Anyone who can reach the server can use the API, and that uses your Anthropic API key and credits. Do not expose it to the public internet as it is.
- **No CORS setup.** It is not needed, because Swagger UI is served by the same app.
- **Sample data.** The restaurants, details and availability in `tools.py` are made-up examples. Some restaurants (for example Pasta Dreams) are in the search results but have no entry in the details data.
- If a request fails, `/chat` returns `502` and the server log has the full error.

## Adding a new tool

Open `restaurant-agent-backend/tools.py`, add a function with the `@tool` decorator, then add it to the `tools` list at the bottom. The docstring is what Claude reads to decide when to use the tool, so describe it clearly. Type hints become the tool's inputs.

```python
@tool
def get_menu(restaurant_name: str) -> list:
    """Get the menu items of a restaurant."""
    menus = {"Spice Palace": ["Butter Chicken", "Biryani"]}
    return menus.get(restaurant_name.strip(), [])


tools = [
    search_restaurants_by_cuisine,
    get_restaurant_details,
    check_reservation_availability,
    get_menu,
]
```

Restart the server (`--reload` does this for you). No change to `main.py` is needed.
