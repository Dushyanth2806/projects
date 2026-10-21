import { z } from "zod";

// ---- Tool output schemas (mirror restaurent-agent/tools.py) ----
export const RestaurantSummary = z.object({
  name: z.string(),
  rating: z.number(),
  price: z.string(),
});

export const RestaurantDetails = z.object({
  cuisine: z.string(),
  rating: z.number(),
  address: z.string(),
  phone: z.string(),
  hours: z.string(),
  speciality: z.string(),
});

export const ToolError = z.object({ error: z.string() });

/** The backend sends every tool result as a string; list/dict results are JSON text. */
const jsonText = <T extends z.ZodTypeAny>(schema: T) =>
  z.preprocess((v) => {
    if (typeof v !== "string") return v;
    try { return JSON.parse(v); } catch { return v; }
  }, schema);

// ---- Tool calls: discriminated union on `name`; the backend sends name + args + output ----
export const ToolCall = z.discriminatedUnion("name", [
  z.object({
    name: z.literal("search_restaurants_by_cuisine"),
    args: z.object({ cuisine_type: z.string() }),
    output: jsonText(z.array(RestaurantSummary)),
  }),
  z.object({
    name: z.literal("get_restaurant_details"),
    args: z.object({ restaurant_name: z.string() }),
    output: jsonText(z.union([RestaurantDetails, ToolError])),
  }),
  z.object({
    name: z.literal("check_reservation_availability"),
    args: z.object({
      restaurant_name: z.string(),
      date: z.string(),
      party_size: z.number(),
    }),
    output: z.string(),
  }),
]);
export type ToolCall = z.infer<typeof ToolCall>;

// ---- Full /chat response (main.py ChatResponse): the Gemini agent's reply and the tool calls it made ----
export const ChatResponse = z.object({
  source: z.string().default("Google Gemini"),
  reply: z.string(),
  tool_calls: z.array(z.unknown()),
});
export type ChatResponse = z.infer<typeof ChatResponse>;

export type ToolResult =
  | { ok: true; call: ToolCall }
  | { ok: false; name: string; raw: unknown; issues: string };

/** Validate each tool call; one that does not match its schema is still shown, as raw JSON. */
export function parseTools(calls: unknown[]): ToolResult[] {
  return calls.map((raw): ToolResult => {
    const parsed = ToolCall.safeParse(raw);
    if (parsed.success) return { ok: true, call: parsed.data };
    const name = (raw as { name?: unknown } | null)?.name;
    return {
      ok: false,
      name: typeof name === "string" ? name : "unknown_tool",
      raw,
      issues: parsed.error.issues.map((i) => `${i.path.join(".") || "(root)"}: ${i.message}`).join("; "),
    };
  });
}
