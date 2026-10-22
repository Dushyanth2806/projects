# Restaurant Assistant API

A restaurant recommendation assistant powered by Claude (Anthropic). Ask for restaurants by cuisine, get details such as address, phone and hours, and check reservation availability. The interface is FastAPI's built-in **Swagger UI** at `/docs`, so there is no separate frontend.

## How it works

```
Swagger UI (/docs)
      |  POST /chat {message, session_id?}
      v
  FastAPI  --- load past turns --->  SESSIONS (in memory, by session_id)
      |
      v
  Claude (Anthropic API)  <--->  tools: search / details / reservation
      |        (loop until Claude gives a final answer)
      v
  save user message + final reply to SESSIONS
      |
      v
  {reply, tool_calls, session_id}
```

Only plain-text turns are saved (your message and the final reply), never tool blocks, so a stored conversation is always valid for the Anthropic API.

## Project structure

```
restaurent-agent-backend/
├── main.py            # FastAPI app, sessions, Claude tool-calling loop
├── tools.py           # Tool functions, sample data, tool schemas
├── requirements.txt   # Python dependencies
├── .env.example       # Environment variable template
├── CODE.md            # Full source listing
└── README.md
```

## Setup

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate    macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env      # Windows: copy .env.example .env
```

Then edit `.env` and set your own `ANTHROPIC_API_KEY`.

## Environment variables

| Variable | Required | Description |
|---|---|---|
| `ANTHROPIC_API_KEY` | Yes | Your Anthropic API key. |
| `MODEL` | No | Claude model to use. Defaults to `claude-haiku-5-5`. |

## Run

```bash
uvicorn main:app --reload --port 8000
```

Open http://localhost:8000 (it redirects to `/docs`).

## Using Swagger UI as the frontend

1. Open http://localhost:8000/docs.
2. Expand `POST /chat` and click **Try it out**.
3. Type a message, for example `Show me Italian restaurants`, and click **Execute**.
4. Copy the `session_id` from the response body.
5. For your next message, paste that value into `session_id` and execute again. The assistant remembers the conversation.
6. To start over, use `DELETE /chat/{session_id}` with the same id.

## API

| Method | Path | Description |
|---|---|---|
| POST | `/chat` | Send a message. Body: `message`, optional `session_id`. Returns `reply`, `tool_calls`, `session_id`. |
| DELETE | `/chat/{session_id}` | Clear a conversation. |
| GET | `/health` | Health check, returns `{"status": "ok"}`. |
| GET | `/` | Redirects to `/docs`. |

## Notes and limits

- Memory is stored in RAM and is lost when the server restarts. It keeps at most 200 sessions (oldest dropped) and 30 messages per session.
- There is no authentication. Anyone who can reach the server can use your API key, so do not expose it publicly.
- CORS is not needed because Swagger UI is served by the same app.
- The restaurant data is sample data in `tools.py`.

## Adding a new tool

1. Write the function in `tools.py`.
2. Add its schema (`name`, `description`, `input_schema`) to `tool_definitions`.
3. Register it in the `tool_functions` dict under the same name.
4. Optionally mention it in the `system_prompt` in `main.py`.

Restart the server and Claude can call it.
