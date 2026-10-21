import "./style.css";
import { ChatResponse, parseTools, type ToolResult } from "./schemas";

/*
 * Backend: restaurent-agent (FastAPI), a Gemini agent with three tools:
 * search by cuisine, restaurant details, reservation availability.
 *   POST /chat   { message, history: [{role, content}] }
 *                -> { source, reply, tool_calls: [{name, args, output}] }
 *   GET  /health -> { status: "ok" }
 */

function $<T extends HTMLElement>(id: string): T {
  const e = document.getElementById(id);
  if (!e) throw new Error(`#${id} missing`);
  return e as T;
}
const chat = $("chat");
const form = $<HTMLFormElement>("form");
const input = $<HTMLInputElement>("input");
const apiUrl = $<HTMLInputElement>("apiUrl");
const health = $("health");
const sendBtn = form.querySelector("button") as HTMLButtonElement;

apiUrl.value = import.meta.env.VITE_API_URL ?? "http://localhost:8000";

type Turn = { role: "user" | "assistant"; content: string };
const history: Turn[] = [];

function el<K extends keyof HTMLElementTagNameMap>(tag: K, cls = "", text?: string): HTMLElementTagNameMap[K] {
  const e = document.createElement(tag);
  if (cls) e.className = cls;
  if (text !== undefined) e.textContent = text;
  return e;
}
const jsonBlock = (v: unknown) => el("pre", "", JSON.stringify(v, null, 2));
const scrollDown = () => { chat.scrollTop = chat.scrollHeight; };
const base = () => apiUrl.value.replace(/\/+$/, "");

// ---------- Backend ----------
async function ask(message: string): Promise<ChatResponse> {
  const res = await fetch(`${base()}/chat`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ message, history }),
  });
  if (!res.ok) {
    const detail = await res.json().then((j: { detail?: unknown }) => (typeof j.detail === "string" ? j.detail : null), () => null);
    throw new Error(detail ?? `Backend returned ${res.status}`);
  }
  return ChatResponse.parse(await res.json());
}

async function checkHealth(): Promise<void> {
  try {
    const r = await fetch(`${base()}/health`);
    const ok = r.ok && ((await r.json()) as { status?: string }).status === "ok";
    health.className = ok ? "dot ok" : "dot bad";
    health.title = ok ? "Backend online" : "Backend unhealthy";
  } catch {
    health.className = "dot bad";
    health.title = "Backend unreachable";
  }
}

// ---------- Rendering ----------
function renderTool(t: ToolResult): HTMLElement {
  const box = el("div", "tool");

  if (!t.ok) {
    box.append(el("div", "tool-name", `🔧 ${t.name}`));
    box.append(el("div", "msg bot err", `Unexpected tool output (${t.issues})`), jsonBlock(t.raw));
    return box;
  }

  const c = t.call;
  const args = Object.entries(c.args).map(([k, v]) => `${k}=${JSON.stringify(v)}`).join(", ");
  box.append(el("div", "tool-name", `🔧 ${c.name}(${args})`));
  const grid = el("div", "cards");

  // Exhaustive switch: adding a tool to the schema without a renderer is a compile error.
  switch (c.name) {
    case "search_restaurants_by_cuisine": {
      if (!c.output.length) { box.append(el("div", "msg bot", "No restaurants found for that cuisine.")); break; }
      for (const r of c.output) {
        const card = el("div", "card");
        card.append(el("h3", "", r.name), el("div", "meta", `⭐ ${r.rating}  ·  ${r.price}`));
        grid.append(card);
      }
      box.append(grid);
      break;
    }
    case "get_restaurant_details": {
      if ("error" in c.output) { box.append(el("div", "msg bot err", c.output.error)); break; }
      const card = el("div", "card");
      const dl = el("dl");
      for (const [k, v] of Object.entries(c.output)) dl.append(el("dt", "", k), el("dd", "", String(v)));
      card.append(dl);
      grid.append(card);
      box.append(grid);
      break;
    }
    case "check_reservation_availability": {
      const ok = c.output.startsWith("✓");
      const card = el("div", "card");
      card.append(el("span", `badge ${ok ? "ok" : "bad"}`, ok ? "Available" : "Unavailable"), el("div", "meta", c.output));
      grid.append(card);
      box.append(grid);
      break;
    }
    default: {
      const _never: never = c;
      void _never;
    }
  }
  return box;
}

/** The agent's answer: any tool results it fetched, then its reply. */
function renderAnswer(r: ChatResponse): HTMLElement {
  const box = el("div", "answer gemini");
  box.append(el("div", "source-tag", r.source));
  parseTools(r.tool_calls).forEach((t) => box.append(renderTool(t)));
  if (r.reply) box.append(el("div", "answer-text", r.reply));
  return box;
}

// ---------- Events ----------
async function send(text: string): Promise<void> {
  text = text.trim();
  if (!text) return;
  chat.append(el("div", "msg user", text));
  input.value = "";
  const typing = el("div", "msg bot typing", "Thinking…");
  chat.append(typing);
  scrollDown();
  sendBtn.disabled = true;
  try {
    const r = await ask(text);
    history.push({ role: "user", content: text }, { role: "assistant", content: r.reply });
    chat.insertBefore(renderAnswer(r), typing);
  } catch (e) {
    const msg = e instanceof Error ? e.message : String(e);
    chat.append(el("div", "msg bot err", `${msg} (is the backend running at ${base()}?)`));
  } finally {
    typing.remove();
    sendBtn.disabled = false;
    input.focus();
    scrollDown();
  }
}

form.addEventListener("submit", (e) => { e.preventDefault(); void send(input.value); });
apiUrl.addEventListener("change", () => void checkHealth());

chat.append(el("div", "msg bot", "Hi! I can find restaurants by cuisine, show details, and check reservation availability."));
void checkHealth();
