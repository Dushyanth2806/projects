# Restaurant Assistant (OpenAI + FastAPI + Swagger UI)

A restaurant recommendation assistant. The backend is a FastAPI app that talks to
the OpenAI API and lets the model call three restaurant tools. There is **no
separate frontend**: you chat with it from the Swagger UI page that FastAPI
serves automatically at `/docs`.

## What it can do

- **Search restaurants by cuisine**: Indian, Italian, Chinese, Mexican.
- **Get restaurant details**: address, phone, hours, specialities.
- **Check reservation availability** for a restaurant, date and party size.

Anything else is answered as "outside my capabilities". The restaurant data is
sample data hard-coded in `restaurant-agent-backend/tools.py`.

## How it works

```
Swagger UI (/docs)  ->  POST /chat  ->  OpenAI model  <->  restaurant tools
                           |
                     session memory (in RAM, keyed by session_id)
```

1. You send a `message` (and optionally a `session_id`).
2. The server loads that session's earlier messages and calls OpenAI with the
   system prompt, the conversation and the tool definitions.
3. If the model asks for a tool, the server runs it and sends the result back.
   This repeats up to 5 times.
4. The final text is returned as `reply`, together with the `tool_calls` that
   were used and the `session_id`.

## Project structure

```
Restaurant-Agent(OpenAI)/
├── README.md
└── restaurant-agent-backend/
    ├── main.py            FastAPI app, session memory, OpenAI tool-calling loop
    ├── tools.py           The three tools + their OpenAI definitions
    ├── requirements.txt   Python dependencies
    ├── .env.example       Template for your environment variables
    └── CODE.md            Full source listing of the backend
```

## Setup

You need Python 3.10+ and an OpenAI API key.

```powershell
cd restaurant-agent-backend

python -m venv .venv
.venv\Scripts\activate          # macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt

copy .env.example .env          # macOS/Linux: cp .env.example .env
# open .env and set OPENAI_API_KEY=sk-...
```

### Environment variables

| Variable | Required | Default | Meaning |
|---|---|---|---|
| `OPENAI_API_KEY` | yes | none | Your OpenAI API key |
| `MODEL` | no | `gpt-4o-mini` | OpenAI model to use |

`.env` is git-ignored. Never commit your real key.

## Run

```powershell
uvicorn main:app --reload --port 8000
```

Then open **http://localhost:8000/docs**. (`http://localhost:8000/` redirects
there.) If port 8000 is busy, use another one, e.g. `--port 8001`.

## Using Swagger UI as the frontend

1. Open http://localhost:8000/docs.
2. Click **POST /chat**, then **Try it out**.
3. In the request body, set a message and leave `session_id` empty (or `null`):
   ```json
   {
     "message": "Show me Italian restaurants"
   }
   ```
4. Click **Execute**. Read the answer under **Response body**:
   ```json
   {
     "reply": "Here are some Italian restaurants ...",
     "tool_calls": [{"name": "search_restaurants_by_cuisine", "output": "[...]"}],
     "session_id": "02ba4a036de54fa99bc2230e7932bf93"
   }
   ```
5. **Copy the `session_id`** and paste it into the request body of your next
   message. The assistant then remembers the conversation:
   ```json
   {
     "message": "What is the phone number of the highest rated one?",
     "session_id": "02ba4a036de54fa99bc2230e7932bf93"
   }
   ```
6. To start over, either leave `session_id` empty, or call
   **DELETE /chat/{session_id}** to forget that conversation.

Tips:
- Reservation dates in the sample data are `2024-11-15`, `2024-11-16` and
  `2024-11-17`.
- `tool_calls` shows which tools ran and their raw output, which is handy for
  checking that the assistant is not making things up.

## API reference

| Method | Path | Purpose |
|---|---|---|
| `POST` | `/chat` | Send a message; returns `reply`, `tool_calls`, `session_id` |
| `DELETE` | `/chat/{session_id}` | Clear one conversation |
| `GET` | `/health` | Returns `{"status": "ok"}` |
| `GET` | `/docs` | Swagger UI |
| `GET` | `/openapi.json` | The machine-readable API description |

### Calling it without Swagger

```powershell
curl -X POST http://localhost:8000/chat -H "Content-Type: application/json" `
  -d '{\"message\": \"Show me Mexican restaurants\"}'
```

## Notes and limits

- **Memory is in RAM.** Sessions are lost when the server restarts. Only the
  last 30 messages of a session and the 200 most recent sessions are kept.
- **No authentication.** Anyone who can reach the server can use your OpenAI
  key through it. Keep it on localhost, or add authentication before deploying.
- **No CORS.** Swagger UI is served by the same app, so it isn't needed. If you
  later add a browser frontend on another origin, add FastAPI's `CORSMiddleware`.
- **Data is sample data.** Replace the dictionaries in `tools.py` with a real
  database or API to make it useful.
- Each request costs OpenAI tokens.

## Adding a new tool

1. Write a function in `tools.py` with a clear docstring.
2. Add it to `TOOL_FUNCTIONS`.
3. Add a `_schema(...)` entry for it in the `tools` list.

The model decides when to call it based on the docstring and parameters.
