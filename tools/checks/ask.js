// Drives the pinned "Ask the model" chat with a fake window.claude "sample" (no real Claude calls) through a
// two-turn conversation: the first answer adds a made-up number (caught and rewritten), the follow-up must
// carry the first exchange as history. Usage: node ask.js ../../web/index.html
const { JSDOM, VirtualConsole } = require("jsdom");
const html = require("fs").readFileSync(process.argv[2], "utf8");
const vc = new VirtualConsole(); let errs = 0;
vc.on("jsdomError", e => { errs++; console.log("ERROR:", e.message); });
const calls = [];
const dom = new JSDOM(html, { runScripts: "dangerously", pretendToBeVisual: true, virtualConsole: vc, url: "https://claude.ai/artifact/x",
  beforeParse(w) {
    w.scrollTo = () => {};
    const sample = async (input, opts = {}) => {
      calls.push({ input, tools: !!opts.tools });
      if (!opts.tools) return { text: "2025 Nebraska is an 87% favorite over 2023 Texas.", truncated: false };   // the rewrite
      const home = Array.isArray(input) && input.length > 2;   // the follow-up: "And at Nebraska?"
      const t = opts.tools.find(t => t.name === "win_probability");
      const r = await t.execute({ team_a: "Nebraska", season_a: 2025, team_b: "Texas", season_b: 2023, venue: home ? "team_a_home" : "neutral" });
      const p = Math.round(100 * r.p_team_a_wins);
      const text = home ? `At home, 2025 Nebraska is a ${p}% favorite.` : `2025 Nebraska is a ${p}% favorite over 2023 Texas, and hit .412.`;
      opts.onText?.({ text, delta: text });
      return { text, truncated: false };
    };
    sample.limits = async () => ({ maxPromptBytes: 262144, tools: { maxCount: 16 } });
    w.claude = { use: async name => (name === "sample" ? sample : null) };
  } });
const w = dom.window, d = w.document, wait = ms => new Promise(r => setTimeout(r, ms));
setTimeout(async () => {
  console.log("launcher on home:", !!d.getElementById("chat-fab"), "| panel hidden:", d.getElementById("chat").hidden);
  w.location.hash = "matchups"; w.dispatchEvent(new w.HashChangeEvent("hashchange"));
  d.getElementById("chat-fab").click();
  console.log("open:", !d.getElementById("chat").hidden, "| first suggestion on simulator:", d.querySelector(".chat-empty .chip").textContent);
  d.getElementById("chat-q").value = "Could Nebraska have beaten 2023 Texas?";
  d.getElementById("chat-form").requestSubmit(); await wait(300);
  d.getElementById("chat-q").value = "And at Nebraska?";
  d.getElementById("chat-form").requestSubmit(); await wait(300);
  const msgs = [...d.querySelectorAll("#chat-log .msg")].map(m => (m.classList.contains("user") ? "you: " : "bot: ") + (m.querySelector(".answer") || m).textContent.trim().slice(0, 70));
  console.log(msgs.join("\n"));
  console.log("statuses:", [...d.querySelectorAll("#chat-log .ask-status")].map(s => s.dataset.state).join(", "));
  const follow = calls.filter(c => c.tools)[1].input;
  console.log("follow-up sent:", follow.map(m => m.role).join(","), "| carries first answer:", follow[2]?.content.includes("87%"),
    "| rewrite named .412:", calls[1].input.includes(".412"));
  d.getElementById("chat-new").click();
  console.log("new chat clears:", d.querySelectorAll("#chat-log .msg").length === 0);
  console.log("errors:", errs); process.exit(errs ? 1 : 0);
}, 1500);
