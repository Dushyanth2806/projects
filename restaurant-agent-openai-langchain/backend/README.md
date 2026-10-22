# Restaurant Assistant API

A chat assistant that recommends restaurants, shows their details and checks reservation availability. It is a FastAPI app with a LangChain agent powered by OpenAI. There is no separate frontend: **FastAPI's Swagger UI at `/docs` is the frontend.**

## How it works

```
 Swagger UI (/docs)
        |  POST /chat  { message, session_id? }
        v
 FastAPI  --- load past messages from SESSIONS[session_id]
        |
        v
 LangChain agent (OpenAI model)
        |  decides which tool to call
        v
 Tools: search by cuisine | restaurant details | reservation availability
        |
        v
 reply + tool_calls + session_id  --> saved back into SESSIONS --> Swagger UI
```

Conversation memory lives on the server, keyed by `session_id`, so you never paste chat history yourself.

## Project structure

```
restaurant-agent-backend/
├── main.py             # FastAPI app, agent setup, endpoints, session memory
├── tools.py            # The three restaurant tools (sample data)
├── requirements.txt    # Python dependencies
├── .env.example        # Template for your .env
├── CODE.md             # Full source listing
└── README.md
```

## Setup

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt

# Windows (cmd): copy .env.example .env
cp .env.example .env
```

Then open `.env` and put your own OpenAI API key in it.

## Environment variables

| Variable | Required | Description |
|----------|----------|-------------|
| `OPENAI_API_KEY` | Yes | Your OpenAI API key |
| `MODEL` | No | Chat model, default `openai:gpt-4o-mini` |

## Run

```bash
uvicorn main:app --reload --port 8000
```

Open http://127.0.0.1:8000 (it redirects to `/docs`).

## Using Swagger UI as the frontend

1. Open http://127.0.0.1:8000/docs.
2. Expand **POST /chat** and click **Try it out**.
3. Type a message in the request body, e.g. `{"message": "Show me Italian restaurants"}`, and click **Execute**.
4. Copy the `session_id` from the response.
5. For your next message, paste that `session_id` into the request body, e.g. `{"message": "Give me details of the best rated one", "session_id": "<your id>"}`, and click **Execute**. The assistant remembers the earlier messages.
6. To start over, expand **DELETE /chat/{session_id}**, enter the id and execute it.

## API

| Method | Path | Description |
|--------|------|-------------|
| GET | `/` | Redirects to `/docs` |
| POST | `/chat` | Body: `message` (required), `session_id` (optional). Returns `reply`, `tool_calls`, `session_id` |
| DELETE | `/chat/{session_id}` | Clears that conversation |
| GET | `/health` | Returns `{"status": "ok"}` |

## Notes and limits

- **Memory is in RAM.** Conversations are lost when the server restarts. At most 200 sessions are kept (the oldest is dropped) and at most 30 messages per session.
- **No authentication.** Anyone who can reach the server can use it, and therefore your OpenAI API key and quota. Don't expose it publicly as is.
- **CORS is not needed.** Swagger is served by the same app, so requests are same-origin.
- **Sample data.** The restaurants, details and reservation slots in `tools.py` are made-up examples.

## Adding a new tool

1. In `tools.py`, write a function decorated with `@tool`. Give it type hints and a clear docstring, because the model reads the docstring to decide when to use it.
2. Add the function to the `tools` list at the bottom of `tools.py`.
3. Optionally mention it in `system_prompt` in `main.py`.
4. Restart the server (or let `--reload` do it) and ask the assistant something that needs the tool.
