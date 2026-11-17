// ReBel customer app: one phone, six views (welcome, consent, chat, reveal, plan, done).
// The server holds the logic (mobile/server.py); this file shows it and sends the customer's answers.

const eur = new Intl.NumberFormat("en-IE", { style: "currency", currency: "EUR", maximumFractionDigits: 0 });
const pct = (weight) => `${Math.round(weight * 100)}%`;
const $ = (id) => document.getElementById(id);
const sum = (items, value) => items.reduce((total, item) => total + value(item), 0);

// The sleeves of a model portfolio: the colour follows the sleeve, never its rank.
const COLOR = { core_equity: "--shares", bonds: "--bonds", gold: "--gold", satellites: "--crypto", cash: "--cash" };
// What a state of an ETF sheet looks like: an arrow and a tone, so that it never rests on colour alone.
const STATE = { up: ["↗", "good"], supportive: ["↗", "good"], neutral: ["→", "wait"], watch: ["!", "wait"], down: ["↘", "bad"], headwind: ["↘", "bad"] };

const svg = (size, body, width = 2) =>
  `<svg width="${size}" height="${size}" viewBox="0 0 20 20" fill="none" stroke="currentColor" stroke-width="${width}" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">${body}</svg>`;
const ICON = {
  back: svg(20, '<path d="M12.5 4.5 7 10l5.5 5.5"/>'),
  up: svg(20, '<path d="M10 15.5v-11M5 9.5l5-5 5 5"/>', 2.4),
  check: svg(16, '<path d="M4 10.5l4 4 8-9"/>', 2.4),
  bigCheck: svg(46, '<path d="M4.5 10.5l3.6 3.8 7.4-8.6"/>', 2.2),
  cross: svg(16, '<path d="M5.5 5.5l9 9M14.5 5.5l-9 9"/>', 2.4),
  alert: svg(16, '<path d="M10 3 2.5 16h15L10 3zM10 8.5v3.5M10 14.4v.1"/>'),
  lock: svg(16, '<rect x="4" y="9" width="12" height="8" rx="2"/><path d="M7 9V6.5a3 3 0 0 1 6 0V9"/>'),
  person: svg(16, '<circle cx="10" cy="7" r="3.2"/><path d="M3.5 17a6.5 6.5 0 0 1 13 0"/>'),
};
const LINE = "M0 150 28 138 52 146 80 118 104 128 132 96 158 110 186 78 212 92 240 58 266 74 296 40 322 56 352 24 390 34";
const HERO_ART = `<svg viewBox="0 0 390 190" preserveAspectRatio="none" fill="none" aria-hidden="true">
  <defs><linearGradient id="fade" x1="0" y1="0" x2="0" y2="1"><stop class="top" offset="0"/><stop class="bottom" offset="1"/></linearGradient>
  <filter id="glow" x="-10%" y="-60%" width="120%" height="220%"><feGaussianBlur stdDeviation="7"/></filter></defs>
  <path d="${LINE} V190 H0Z" fill="url(#fade)"/>
  <path class="glow" d="${LINE}" filter="url(#glow)"/>
  <path class="stroke" d="${LINE}"/></svg>`;

const blank = () => ({ consent: true, answers: {}, question: null, picked: [], odds: null, seen: [], sheet: [], flags: [],
  left: null, profile: null, why: "", decision: null, decided_at: null, plan: null, busy: false });
const state = { customers: [], profiles: {}, sleeves: {}, questions: 0, accountData: [], coach: "", id: null, known: {}, ...blank() };
let run = 0; // grows when the customer changes, so a late answer for the previous one is dropped

/** Builds an element. Text children are set as text, never as HTML: what Claude or the customer wrote stays text. */
function h(tag, attrs = {}, ...children) {
  const el = document.createElement(tag);
  for (const [key, value] of Object.entries(attrs)) {
    if (value == null || value === false) continue;
    if (key === "class") el.className = value;
    else if (key === "icon") el.innerHTML = value; // only ever one of the ICON strings above
    else if (key.startsWith("on")) el.addEventListener(key.slice(2), value);
    else el.setAttribute(key, value === true ? "" : value);
  }
  for (const child of children.flat()) {
    if (child == null || child === false) continue;
    el.append(child.nodeType ? child : document.createTextNode(String(child)));
  }
  return el;
}

async function api(path, body) {
  const response = await fetch(path, body
    ? { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) } : undefined);
  const data = await response.json().catch(() => ({}));
  if (!response.ok) throw new Error(data.error || `The request failed (${response.status}).`);
  return data;
}

const me = () => state.customers.find((c) => c.id === state.id);
const profile = () => state.profiles[state.profile];
const avatar = (c) => h("span", { class: "avatar" }, c.first_name[0]);
const when = (iso) => new Date(iso).toLocaleString("en-GB", { day: "numeric", month: "short", hour: "2-digit", minute: "2-digit" });

function go(name) {
  closeSheet();
  for (const view of document.querySelectorAll(".view")) view.classList.toggle("is-active", view.id === `view-${name}`);
}

// ---- Bottom sheet -------------------------------------------------------------------------------------------
function openSheet(...children) {
  $("sheet").replaceChildren(...children);
  $("sheet").hidden = $("backdrop").hidden = false;
}
function closeSheet() {
  $("sheet").hidden = $("backdrop").hidden = true;
}

// ---- Welcome ------------------------------------------------------------------------------------------------
function renderWelcome() {
  const c = me();
  const back = Boolean(state.profile);
  const status = state.decision === "approved" ? "is approved" : "is waiting for your approval";
  $("view-welcome").replaceChildren(
    h("div", { class: "welcome-top" },
      h("p", { class: "wordmark" }, "ReBel"),
      h("h2", { class: "display welcome-title" },
        ...(back ? ["Welcome back, ", h("em", {}, `${c.first_name}.`)] : ["Investing that fits ", h("em", {}, "you.")])),
      h("p", { class: "welcome-sub" }, back ? `Your ${profile().label} plan ${status}.`
        : `Three minutes with ${state.coach}. A plan built for your life. One tap to approve it.`)),
    h("div", { class: "hero-art", icon: HERO_ART }),
    h("div", { class: "grow" }),
    h("div", { class: "welcome-bottom" },
      h("div", { class: "signed" }, avatar(c),
        h("div", {}, h("b", {}, `${c.first_name}, ${c.age}`), h("small", {}, `Signed in · ${c.city}`))),
      h("button", { class: "btn btn-primary", type: "button", onclick: back ? openPlan : () => go("consent") },
        back ? "See my plan" : "Get started")));
}

// ---- Consent ------------------------------------------------------------------------------------------------
function renderConsent() {
  const items = state.accountData.map((label) => h("li", {}, h("span", {}, label), h("b", {}, state.known[label])));
  $("view-consent").replaceChildren(
    h("div", { class: "topbar" }, h("button", { class: "icon-btn", type: "button", "aria-label": "Back", icon: ICON.back, onclick: () => go("welcome") })),
    h("div", { class: "pad scroll" },
      h("h2", { class: "display screen-title" }, "Skip what we already know."),
      h("p", { class: "lead" }, `With your OK, ${state.coach} reads these five things from your Belfius accounts and only asks what is missing.`),
      h("ul", { class: "data-list" }, items),
      h("p", { class: "notice" }, h("span", { icon: ICON.person }),
        `${state.coach} is an AI. A Belfius adviser is one tap away, at any time.`)),
    h("div", { class: "footer" },
      h("button", { class: "btn btn-primary", type: "button", onclick: () => startChat(true) }, "Use my data"),
      h("button", { class: "btn btn-text", type: "button", onclick: () => startChat(false) }, "I'd rather answer myself")));
}

// ---- Chat ---------------------------------------------------------------------------------------------------
function renderChat() {
  $("view-chat").replaceChildren(
    h("div", { class: "chat-head" }, h("div", { class: "orb" }),
      h("div", { class: "grow" }, h("b", {}, state.coach), h("small", {}, "AI · ReBel")),
      h("button", { class: "pill", type: "button", onclick: adviserSheet }, "Adviser")),
    h("div", { class: "progress" }, h("i", { id: "progress" })),
    h("div", { class: "messages", id: "messages", "aria-live": "polite" }),
    h("div", { class: "chips", id: "chips" }),
    h("form", { class: "composer", onsubmit: (event) => { event.preventDefault(); typed($("input").value); } },
      h("input", { id: "input", type: "text", placeholder: "Or type your answer", autocomplete: "off", maxlength: "400", "aria-label": "Your answer" }),
      h("button", { class: "send", id: "send", type: "submit", "aria-label": "Send", icon: ICON.up })));
}

function addToChat(el) {
  $("messages").append(el);
  $("messages").scrollTo({ top: $("messages").scrollHeight, behavior: "smooth" });
}
const addBubble = (role, text) => addToChat(h("div", { class: `bubble ${role} pop` }, text));

function setTyping(on) {
  $("typing")?.remove();
  if (on) addToChat(h("div", { class: "typing pop", id: "typing", "aria-label": `${state.coach} is writing` }, h("i"), h("i"), h("i")));
}

/** The answers to tap for the question asked. The customer's rules take several: tap them, then confirm. */
function showChoices() {
  const question = state.question;
  const chip = (label, onclick, on) => h("button", { class: on ? "chip pop is-on" : "chip pop", type: "button", onclick }, label);
  let chips = [];
  if (question?.kind === "rules") {
    const toggle = (value) => { state.picked = state.picked.includes(value) ? state.picked.filter((v) => v !== value) : [...state.picked, value]; showChoices(); };
    chips = [...question.choices.map((c) => chip(c.label, () => toggle(c.value), state.picked.includes(c.value))),
      chip(state.picked.length ? "That is all" : "No, nothing", () => answer(state.picked,
        state.picked.length ? question.choices.filter((c) => state.picked.includes(c.value)).map((c) => c.label).join(", ") : "No, nothing"))];
  } else if (question) {
    chips = question.choices.map((c) => chip(c.label, () => answer(c.value, c.label)));
  }
  $("chips").replaceChildren(...chips);
  $("messages").scrollTop = $("messages").scrollHeight;
}

function startChat(consent) {
  Object.assign(state, blank(), { consent });
  renderChat();
  go("chat");
  turn();
}

const answer = (value, label) => { if (!state.busy) { addBubble("customer", label); turn({ question: state.question.id, value }); } };

function typed(text) {
  text = text.trim();
  if (!text || state.busy || !state.question) return;
  addBubble("customer", text);
  $("input").value = "";
  turn({ question: state.question.id, text });
}

/** Sends the customer's reply, then shows what the coach says, the next question and what the code now holds. */
async function turn(reply) {
  const mine = run;
  state.busy = true;
  $("chips").replaceChildren();
  setTyping(true);
  sync();
  try {
    const next = await api("/api/chat", { customer: state.id, consent: state.consent, answers: state.answers, reply });
    if (mine !== run) return;
    Object.assign(state, { answers: next.answers, question: next.question, picked: [], odds: next.odds, seen: next.seen,
      sheet: next.sheet, flags: next.flags, left: next.left, profile: next.profile, why: next.why });
    setTyping(false);
    for (const line of next.said) addBubble("coach", line);
    if (next.question) {
      addBubble("coach", next.question.text);
    } else {
      addToChat(h("div", { class: "ready pop" }, h("b", {}, "Your profile is ready"),
        h("p", {}, "I found the one that fits your answers, and the plan that goes with it."),
        h("button", { class: "btn btn-primary", type: "button", onclick: () => { renderReveal(); go("reveal"); } }, "Show me")));
    }
  } catch (error) {
    if (mine !== run) return;
    setTyping(false);
    addToChat(h("div", { class: "chat-error pop" }, h("span", {}, error.message)));
  }
  state.busy = false;
  showChoices();
  sync();
}

function adviserSheet() {
  openSheet(h("h3", {}, "Talk to a person"),
    h("p", {}, `A Belfius adviser can take over at any point, with what you already told ${state.coach}, so you never start again.`),
    h("p", {}, "This is a prototype: no call is placed."),
    h("button", { class: "btn btn-ghost", type: "button", onclick: closeSheet }, "Close"));
}

// ---- Profile reveal -----------------------------------------------------------------------------------------
const section = (title, ...children) => h("div", { class: "section" }, h("p", { class: "eyebrow" }, title), ...children);
const ticks = (items, icon = ICON.check) => h("ul", { class: "ticks" }, items.map((text) => h("li", {}, h("span", { icon }), text)));
const tile = (label, value, wide) => h("div", { class: wide ? "tile wide" : "tile" }, h("small", {}, label), h("b", {}, value));
const flagsPanel = () => h("div", { class: "panel is-warn" },
  h("b", {}, "An adviser looks at this with you first"), ticks(state.flags, ICON.alert));

function renderReveal() {
  const p = profile();
  $("view-reveal").replaceChildren(
    h("div", { class: "scroll" },
      h("div", { class: `reveal-hero p-${state.profile}` },
        h("p", { class: "eyebrow" }, "Your investor profile"), h("h2", { class: "display" }, p.label), h("p", {}, p.tagline)),
      h("div", { class: "tiles" }, tile("Horizon", p.horizon), tile("Risk level", `${p.level} of 5`), tile("Style", p.risk, true)),
      section("Why this one", h("div", { class: "panel" }, h("p", {}, state.why))),
      state.flags.length ? section("Before you invest", flagsPanel()) : null,
      section("How your money is handled", h("div", { class: "panel" }, ticks(p.rules))),
      h("div", { style: "height:14px" })),
    h("div", { class: "footer" },
      h("button", { class: "btn btn-primary", type: "button", onclick: openPlan }, "See my plan"),
      h("button", { class: "btn btn-text", type: "button", onclick: startOver }, "Redo the questions")));
}

// ---- Plan ---------------------------------------------------------------------------------------------------
async function openPlan() {
  state.plan = await api(`/api/plan?customer=${state.id}`);
  renderPlan();
  go("plan");
}

/** One whole split into its sleeves: a 28px bar with the share printed in each block wide enough for it. */
function allocation(lines, amount) {
  const groups = Object.entries(state.sleeves).map(([key, name]) => ({ key, name, color: COLOR[key],
    weight: sum(lines.filter((l) => l.sleeve === key), (l) => l.weight) })).filter((group) => group.weight > 0);
  const tip = h("div", { class: "tip", hidden: true });
  const bar = h("div", { class: "bar", role: "img", "aria-label": groups.map((g) => `${g.name} ${pct(g.weight)}`).join(", ") });
  for (const group of groups) {
    const part = h("div", { class: "part", tabindex: "0", style: `flex:${group.weight} 0 0;background:var(${group.color});color:var(${group.color.replace("--", "--on-")})`,
      "aria-label": `${group.name}: ${pct(group.weight)}, ${eur.format(amount * group.weight)}` }, group.weight >= 0.1 ? pct(group.weight) : "");
    const show = () => {
      tip.replaceChildren(h("b", {}, group.name), ` · ${pct(group.weight)} · ${eur.format(amount * group.weight)}`);
      tip.hidden = false;
      const middle = part.offsetLeft + part.offsetWidth / 2;
      tip.style.left = `${Math.min(Math.max(middle, tip.offsetWidth / 2), bar.offsetWidth - tip.offsetWidth / 2)}px`;
    };
    const hide = () => { tip.hidden = true; };
    for (const [event, action] of [["mouseenter", show], ["focus", show], ["mouseleave", hide], ["blur", hide]]) part.addEventListener(event, action);
    bar.append(part);
  }
  const legend = h("ul", { class: "legend" }, groups.map((g) => h("li", {}, h("i", { style: `background:var(${g.color})` }), g.name,
    h("span", {}, g.weight >= 0.1 ? eur.format(amount * g.weight) : `${pct(g.weight)} · ${eur.format(amount * g.weight)}`))));
  return h("div", { class: "alloc" }, tip, bar, legend);
}

function initials(ticker) {
  const words = ticker.replace(/[^A-Za-z0-9 ]/g, "").split(" ").filter(Boolean);
  return (words.length > 1 ? words[0][0] + words[1][0] : words[0].slice(0, 2)).toUpperCase();
}

function lineRow(line) {
  return h("button", { class: "line", type: "button", onclick: () => openEtf(line) },
    h("span", { class: "token" }, initials(line.ticker), h("i", { style: `background:var(${COLOR[line.sleeve]})` })),
    h("span", { class: "name" }, h("b", {}, line.ticker), h("small", {}, line.theme)),
    h("span", { class: "figures" }, h("b", {}, pct(line.weight)), h("small", {}, eur.format(line.eur))));
}

function sleeve(lines, key) {
  const mine = lines.filter((line) => line.sleeve === key);
  if (!mine.length) return null;
  return h("div", { class: "section" },
    h("div", { class: "sleeve-head" }, h("b", {}, `${state.sleeves[key]} · ${pct(sum(mine, (l) => l.weight))}`),
      h("small", {}, mine.length > 1 ? `${mine.length} ETFs` : "1 ETF")),
    h("div", { class: "lines" }, mine.map(lineRow)));
}

// ---- ETF sheet ----------------------------------------------------------------------------------------------
const shownValue = (line) => (line.block === "returns" ? `${line.value >= 0 && line.name !== "volatility" ? "+" : ""}${(line.value * 100).toFixed(1)}%` : line.state);

/** Opens the sheet of one ETF of the plan: what it did, where its trend stands, what macro means for it. */
async function openEtf(line) {
  let sheet = null;
  try { sheet = await api(`/api/etf?ticker=${encodeURIComponent(line.ticker)}`); } catch (error) { sheet = { missing: error.message }; }
  renderEtf(line, sheet);
  go("etf");
}

function renderEtf(line, sheet) {
  const detail = h("div", { class: "section", id: "tile-detail" });
  const pick = (item) => {
    const checks = item.detail.split("; ").filter(Boolean);
    const answer = h("p", { class: "why", hidden: true });
    const ask = h("button", { class: "btn btn-ghost", type: "button" }, `Ask ${state.coach} why`);
    ask.addEventListener("click", async () => {
      ask.disabled = true;
      ask.textContent = `${state.coach} is reading the sheet...`;
      try { answer.textContent = (await api("/api/why", { ticker: line.ticker, block: item.block, name: item.name })).answer; } catch (error) { answer.textContent = error.message; }
      answer.hidden = false;
      ask.remove();
    });
    detail.replaceChildren(h("p", { class: "eyebrow" }, `${item.block} · ${item.name}`),
      h("div", { class: "panel" }, h("ul", { class: "ticks" }, (checks.length ? checks : [shownValue(item)]).map((text) =>
        h("li", { class: text.includes(": no") ? "is-no" : "" }, h("span", { icon: text.includes(": no") ? ICON.cross : ICON.check }), text))), answer, ask));
    detail.scrollIntoView({ behavior: "smooth", block: "nearest" });
  };
  const tiles = (block, title) => {
    const items = (sheet.lines || []).filter((l) => l.block === block);
    if (!items.length) return null;
    return section(title, h("div", { class: "sheet-tiles" }, items.map((item) => {
      const [arrow, tone] = STATE[item.state] || ["", "plain"];
      return h(block === "returns" ? "div" : "button", { class: `sheet-tile is-${tone}`, type: block === "returns" ? null : "button", onclick: block === "returns" ? null : () => pick(item) },
        h("small", {}, item.name), h("b", {}, arrow ? `${arrow} ${item.state}` : shownValue(item)));
    })));
  };
  $("view-etf").replaceChildren(
    h("div", { class: "topbar" },
      h("button", { class: "icon-btn", type: "button", "aria-label": "Back to my plan", icon: ICON.back, onclick: () => go("plan") }),
      h("b", {}, line.ticker), h("span", { style: "width:40px" })),
    h("div", { class: "scroll" },
      h("div", { class: "amount" }, h("span", { class: "badge profile" }, state.sleeves[line.sleeve]),
        h("h2", { class: "display etf-name" }, line.name),
        h("p", {}, `${pct(line.weight)} of your plan · ${eur.format(line.eur)} · ISIN ${line.isin}`)),
      sheet.missing ? h("p", { class: "note" }, sheet.missing) : null,
      tiles("returns", "Returns"), tiles("trend", "Trend · tap a tile"), tiles("macro", "Macro · tap a tile"), detail,
      sheet.as_of ? h("p", { class: "quiet center" }, `Computed by the morning run of ${sheet.as_of}. Information, not advice.`) : null,
      h("div", { style: "height:14px" })));
}

function planFooter() {
  if (state.decision === "approved") {
    return h("div", { class: "footer" },
      h("div", { class: "decided is-approved" }, h("span", { icon: ICON.check }), `Approved · ${when(state.decided_at)}`),
      h("button", { class: "btn btn-text", type: "button", onclick: startOver }, "Start over"));
  }
  const note = state.decision === "declined" ? `You said not now on ${when(state.decided_at)}.`
    : state.flags.length ? "An adviser confirms the points flagged before anything is bought."
    : "Nothing is invested before you approve.";
  return h("div", { class: "footer" }, h("small", {}, note),
    h("button", { class: "btn btn-primary", type: "button", onclick: approveSheet }, "Approve plan"),
    h("button", { class: "btn btn-text", type: "button", onclick: () => decide("declined") }, "Not now"));
}

function renderPlan() {
  const { lines, amount } = state.plan;
  const monthly = state.answers.monthly;
  $("view-plan").replaceChildren(
    h("div", { class: "topbar" },
      h("button", { class: "icon-btn", type: "button", "aria-label": "Back to my profile", icon: ICON.back, onclick: () => { renderReveal(); go("reveal"); } }),
      h("b", {}, "Your plan"), h("span", { style: "width:40px" })),
    h("div", { class: "scroll" },
      h("div", { class: "amount" }, h("span", { class: "badge profile" }, `${profile().label} · risk level ${profile().level} of 5`),
        h("div", { class: "hero-number" }, eur.format(amount)),
        h("p", {}, monthly ? `to invest now · then ${eur.format(monthly)} a month` : "to invest now")),
      section("Where it goes", h("div", { class: "panel" }, allocation(lines, amount))),
      ...Object.keys(state.sleeves).map((key) => sleeve(lines, key)),
      state.plan.left_out.length ? section("Your rules", h("div", { class: "panel" },
        ticks(state.plan.left_out.map((out) => `No ${out.rule.toLowerCase()}: ${out.ticker} is left out, its ${pct(out.weight)} goes to ${lines[0].ticker}`)))) : null,
      state.flags.length ? section("Before you invest", flagsPanel()) : null,
      section("How your money is handled", h("div", { class: "panel" }, ticks(profile().rules))),
      h("div", { style: "height:14px" })),
    planFooter());
}

function approveSheet() {
  const { lines, amount } = state.plan;
  openSheet(h("h3", {}, "Approve this plan?"),
    h("p", {}, `${eur.format(amount)} across ${lines.length} ETFs, as shown. Nothing is bought without this approval.`),
    h("div", { class: "stack" },
      h("button", { class: "btn btn-primary", type: "button", onclick: () => decide("approved") }, "Approve"),
      h("button", { class: "btn btn-text", type: "button", onclick: closeSheet }, "Cancel")));
}

async function decide(decision) {
  const { state: now } = await api("/api/decision", { customer: state.id, decision });
  Object.assign(state, { decision: now.decision, decided_at: now.decided_at });
  renderPlan();
  renderWelcome();
  if (decision === "approved") {
    renderDone();
    go("done");
  } else {
    closeSheet();
  }
  sync();
}

// ---- Done ---------------------------------------------------------------------------------------------------
function renderDone() {
  $("view-done").replaceChildren(
    h("div", { class: "done-check", icon: ICON.bigCheck }),
    h("h2", { class: "display done-title" }, "Plan approved."),
    h("p", { class: "done-body" }, "Your approval is recorded. Your orders go out on the next trading day."),
    h("div", { class: "done-actions" }, h("button", { class: "btn btn-ghost", type: "button", onclick: () => go("plan") }, "Back to my plan")));
}

// ---- The presenter's side panels ---------------------------------------------------------------------------
const card = (title, aside, ...children) => h("div", { class: "card" }, h("h3", {}, title, aside ? h("span", {}, aside) : null), ...children);
const rowsOf = (pairs) => h("div", { class: "rows" }, pairs.map(([k, v]) => h("div", {}, h("span", { class: "k" }, k), h("span", { class: "v" }, v))));

/** The probability of each profile: one bar each, in the same order from start to end, the leader in colour. */
function oddsCard() {
  if (!state.odds) return card("Probability of each profile", null, h("p", { class: "quiet" }, "Starts with the first question."));
  const top = Math.max(...Object.values(state.odds));
  const bars = Object.entries(state.odds).map(([key, p]) => h("div", { class: p === top ? "odd is-top" : "odd" },
    h("span", { class: "k" }, state.profiles[key].label), h("span", { class: "track" }, h("i", { style: `width:${Math.max(p * 100, 1)}%` })),
    h("span", { class: "v" }, pct(p))));
  const from = state.seen.map(([name, value]) => `${name}: ${value}`).join(" · ");
  return card("Probability of each profile", state.profile ? "final" : "after each answer", h("div", { class: "odds" }, bars),
    h("p", { class: "quiet" }, `Starts from: ${from}`));
}

/** What the code holds right now: the probabilities, the suitability sheet and the contradictions. */
function sync() {
  const asked = state.left == null ? 0 : state.questions - state.left;
  if ($("progress")) $("progress").style.width = `${Math.round((asked / state.questions) * 100)}%`;
  if ($("send")) $("send").disabled = state.busy || !state.question;
  if ($("input")) $("input").placeholder = state.profile ? "Your profile is ready" : "Or type your answer";
  $("inspector").replaceChildren(
    oddsCard(),
    card("Suitability sheet", state.sheet.length ? `${state.sheet.length} answers` : null,
      state.sheet.length ? rowsOf(state.sheet) : h("p", { class: "quiet" }, "Filled by the same questions.")),
    card("Checks by code", null, state.flags.length
      ? state.flags.map((flag) => h("div", { class: "flag" }, h("span", { icon: ICON.alert }), flag))
      : h("div", { class: "ok-line" }, h("span", { icon: ICON.check }), "No contradiction found")));
  $("people").replaceChildren(...state.customers.map((c) =>
    h("button", { class: c.id === state.id ? "person is-on" : "person", type: "button", onclick: () => selectCustomer(c.id) },
      avatar(c), h("span", {}, h("b", {}, `${c.first_name}, ${c.age}`), h("small", {}, c.status)))));
}

// ---- Customers ----------------------------------------------------------------------------------------------
async function selectCustomer(id) {
  run += 1;
  const { state: saved, known } = await api(`/api/state?customer=${id}`);
  Object.assign(state, blank(), { id, known }, saved ? { ...saved, left: 0 } : {});
  try { localStorage.setItem("rebel.customer", id); } catch { /* private window: the choice is not kept */ }
  renderWelcome();
  renderConsent();
  go("welcome");
  sync();
}

async function startOver() {
  await api("/api/reset", { customer: state.id });
  await selectCustomer(state.id);
}

async function boot() {
  const start = await api("/api/start");
  Object.assign(state, { customers: start.customers, profiles: start.profiles, questions: start.questions,
    accountData: start.account_data, coach: start.coach, sleeves: start.sleeves });
  let last = null;
  try { last = localStorage.getItem("rebel.customer"); } catch { /* private window */ }
  await selectCustomer(state.customers.some((c) => c.id === last) ? last : state.customers[0].id);
  const tick = () => { $("clock").textContent = new Date().toLocaleTimeString("en-GB", { hour: "numeric", minute: "2-digit" }); };
  tick();
  setInterval(tick, 30_000);
}

$("backdrop").addEventListener("click", closeSheet);
$("start-over").addEventListener("click", startOver);
document.addEventListener("keydown", (event) => { if (event.key === "Escape") closeSheet(); });
boot();
