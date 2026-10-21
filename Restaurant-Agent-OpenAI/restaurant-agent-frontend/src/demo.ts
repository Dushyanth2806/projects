/** Offline stand-in for POST /chat. Returns exactly what the real backend returns (tool outputs as strings). */

const RESTAURANTS: Record<string, Array<[string, number, string]>> = {
  indian: [["Spice Palace", 4.5, "₹₹"], ["Curry House", 4.3, "₹₹"], ["Taj Flavors", 4.7, "₹₹₹"]],
  italian: [["Pizza Bella", 4.4, "₹₹"], ["Pasta Dreams", 4.6, "₹₹₹"], ["Roma Kitchen", 4.2, "₹₹"]],
  chinese: [["Dragon House", 4.5, "₹₹"], ["Ming's Kitchen", 4.3, "₹₹"], ["Golden Wok", 4.8, "₹₹₹"]],
  mexican: [["Taco Fiesta", 4.4, "₹₹"], ["Salsa Nights", 4.6, "₹₹₹"], ["Burrito House", 4.2, "₹₹"]],
};

const DETAILS: Record<string, Record<string, string | number>> = {
  "Spice Palace": { cuisine: "Indian", rating: 4.5, address: "123 Food Street, City Center", phone: "9876543210", hours: "11 AM - 11 PM", speciality: "Butter Chicken, Biryani" },
  "Pizza Bella": { cuisine: "Italian", rating: 4.4, address: "456 Dine Avenue, Downtown", phone: "9123456789", hours: "12 PM - 10:30 PM", speciality: "Margherita Pizza, Tiramisu" },
  "Dragon House": { cuisine: "Chinese", rating: 4.5, address: "789 Noodle Lane, Market Area", phone: "9111222333", hours: "11:30 AM - 11 PM", speciality: "Hakka Noodles, Fried Rice" },
  "Taco Fiesta": { cuisine: "Mexican", rating: 4.4, address: "321 Spice Row, Hip District", phone: "9222333444", hours: "1 PM - 12 AM", speciality: "Fish Tacos, Guacamole" },
  "Curry House": { cuisine: "Indian", rating: 4.3, address: "555 Flavor Street, Old Town", phone: "9333444555", hours: "11 AM - 10 PM", speciality: "Paneer Tikka, Dal Makhani" },
  "Taj Flavors": { cuisine: "Indian", rating: 4.7, address: "999 Taste Lane, Premium Zone", phone: "9444555666", hours: "12 PM - 11 PM", speciality: "Biryani, Tandoori Chicken" },
  "Golden Wok": { cuisine: "Chinese", rating: 4.8, address: "777 Wok Street, Central Market", phone: "9555666777", hours: "11 AM - 11 PM", speciality: "Szechuan Noodles, Dim Sum" },
};

const DATES = ["2024-11-15", "2024-11-16", "2024-11-17"];
const SLOTS: Record<string, boolean[]> = {
  "Spice Palace": [true, false, true], "Pizza Bella": [false, true, true], "Dragon House": [true, true, false],
  "Taco Fiesta": [true, true, true], "Curry House": [true, false, true], "Taj Flavors": [false, true, true],
};

export function demoChat(msg: string): unknown {
  const lower = msg.toLowerCase();
  const name = Object.keys(DETAILS).find((n) => lower.includes(n.toLowerCase()));
  const date = msg.match(/\d{4}-\d{2}-\d{2}/)?.[0];
  const party = Number(lower.match(/(\d+)\s*(people|persons|guests|pax)/)?.[1] ?? 2);

  if (name && date) {
    const slots = SLOTS[name];
    const ok = !!slots?.[DATES.indexOf(date)];
    const output = !slots ? `Restaurant '${name}' not found.`
      : ok ? `✓ ${name} has availability for ${party} people on ${date}`
           : `✗ ${name} is fully booked for ${date}`;
    return {
      reply: ok ? `Good news — ${name} has a table for ${party} on ${date}.` : `Sorry, ${name} is fully booked on ${date}.`,
      tool_calls: [{ name: "check_reservation_availability", output }],
    };
  }
  if (name) {
    return {
      reply: `Here are the details for ${name}.`,
      tool_calls: [{ name: "get_restaurant_details", output: JSON.stringify(DETAILS[name]) }],
    };
  }
  const cuisine = Object.keys(RESTAURANTS).find((c) => lower.includes(c));
  if (cuisine) {
    const list = RESTAURANTS[cuisine]!.map(([n, rating, price]) => ({ name: n, rating, price }));
    return {
      reply: `Here are some ${cuisine} restaurants you might like.`,
      tool_calls: [{ name: "search_restaurants_by_cuisine", output: JSON.stringify(list) }],
    };
  }
  return {
    reply: "I can search by cuisine (Indian, Italian, Chinese, Mexican), show restaurant details, or check reservations. The requested information is outside my capabilities otherwise.",
    tool_calls: [],
  };
}
