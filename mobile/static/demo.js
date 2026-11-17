// The online demo has no server: this file answers the app's requests in the browser. It is loaded only by the demo's
// page, which mobile/demo.py writes; the real app (mobile/server.py) does not use it.
// - Lena's logic is the same Python file as in the real app (agents/profiler.py), run in the browser by Pyodide.
// - What the server computes (customers, plans, ETF sheets) is read from demo.json, prepared by mobile/demo.py.
// - There is no model and nothing is sent anywhere: a typed answer that is not an amount, and "Ask Lena why", say so.
(() => {
  const PYODIDE = "https://cdn.jsdelivr.net/pyodide/v0.27.7/full/";
  const prepared = fetch("demo.json").then((reply) => reply.json());
  const saved = {};  // customer -> where they stand, what the server reads back from its customer log

  const lena = (async () => {
    await new Promise((loaded, failed) => {
      const script = Object.assign(document.createElement("script"), { src: `${PYODIDE}pyodide.js`, onload: loaded, onerror: failed });
      document.head.append(script);
    });
    const python = await loadPyodide({ indexURL: PYODIDE });
    for (const file of (await prepared).python) {
      python.FS.mkdirTree(`/home/pyodide/${file.split("/")[0]}`);
      python.FS.writeFile(`/home/pyodide/${file}`, await (await fetch(`py/${file}`)).text());
    }
    python.runPython(`
import json
from agents import profiler
from mobile.customers import CUSTOMERS

def chat(body):
    body = json.loads(body)
    return json.dumps(profiler.step(CUSTOMERS[body["customer"]], bool(body.get("consent")), body.get("answers"), body.get("reply")))
`);
    return python.globals.get("chat");
  })();
  lena.catch(() => {});  // said when the chat needs her

  const problem = (status, message) => Object.assign(new Error(message), { status });

  const routes = {
    "/api/start": async () => (await prepared).start,
    "/api/state": async ({ customer }) => ({ state: saved[customer] || null, known: (await prepared).known[customer] }),
    "/api/chat": async (_, body) => {
      const chat = await lena.catch(() => null);
      if (!chat) throw problem(502, "The demo could not load Lena: check your connection and reload the page.");
      let turn;
      try {
        turn = JSON.parse(chat(JSON.stringify(body)));
      } catch {  // a typed reply that only the model can read
        throw problem(502, "Lena cannot read typed answers in this online demo: tap one instead.");
      }
      if (turn.profile) {
        saved[body.customer] = { profile: turn.profile, why: turn.why, answers: turn.answers, sheet: turn.sheet,
          flags: turn.flags, decision: null, decided_at: null };
      }
      return turn;
    },
    "/api/plan": async ({ customer }) => {
      const now = saved[customer];
      if (!now) throw problem(404, "this customer has no profile yet");
      const { rules, plans } = await prepared;
      const plan = plans[now.profile][rules.filter((rule) => (now.answers.rules || []).includes(rule)).join("+")];
      const amount = Number(now.answers.amount || 0);
      return { ...plan, amount, lines: plan.lines.map((line) => ({ ...line, eur: Math.round(amount * line.weight * 100) / 100 })) };
    },
    "/api/etf": async ({ ticker }) => {
      const sheet = (await prepared).sheets[ticker];
      if (!sheet) throw problem(404, "No sheet for this ETF in the online demo.");
      return sheet;
    },
    "/api/why": async () => {
      throw problem(502, "Lena explains each line in the full app. This online demo runs without the AI model.");
    },
    "/api/decision": async (_, { customer, decision }) => {
      if (!saved[customer] || !["approved", "declined"].includes(decision)) throw problem(400, "no plan to decide on");
      Object.assign(saved[customer], { decision, decided_at: new Date().toISOString().slice(0, 19) });
      return { state: saved[customer] };
    },
    "/api/reset": async (_, { customer }) => {
      delete saved[customer];
      return { state: null };
    },
  };

  const network = window.fetch.bind(window);
  window.fetch = async (input, options) => {
    const url = new URL(typeof input === "string" ? input : input.url, location.href);
    const route = routes[url.pathname];
    if (!route) return network(input, options);
    const reply = (status, data) => new Response(JSON.stringify(data), { status, headers: { "Content-Type": "application/json" } });
    try {
      return reply(200, await route(Object.fromEntries(url.searchParams), options && options.body ? JSON.parse(options.body) : {}));
    } catch (error) {
      return reply(error.status || 500, { error: error.message });
    }
  };
})();
