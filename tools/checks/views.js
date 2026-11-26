const { JSDOM, VirtualConsole } = require("jsdom");
const html = require("fs").readFileSync(process.argv[2], "utf8");
const vc = new VirtualConsole(); let errs = 0;
vc.on("jsdomError", e => { errs++; console.log("ERROR:", e.message); });
vc.on("error", (...a) => { errs++; console.log("console.error:", ...a.map(x => x && x.stack ? x.stack.split("\n").slice(0, 3).join(" | ") : x)); });
const dom = new JSDOM(html, { runScripts: "dangerously", pretendToBeVisual: true, virtualConsole: vc, url: "file:///page/index.html" });
const w = dom.window, d = w.document; w.scrollTo = () => {};
const rows = id => d.querySelectorAll(`#${id} tr, #${id} .line, #${id} li, #${id} .wstep`).length;
setTimeout(() => {
  for (const v of ["bracket", "matchups", "findings", "model", "home"]) {
    w.location.hash = v; w.dispatchEvent(new w.HashChangeEvent("hashchange"));
    const vis = [...d.querySelectorAll(".view")].filter(e => !e.hidden).map(e => e.dataset.view);
    console.log(`#${v} -> visible: ${vis}`);
  }
  console.log("odds rows", rows("odds"), "| upsets", rows("upsets"), "| by-round", rows("by-round"), "| worked steps", rows("worked"),
    "| params rows", rows("params"), "| elite", rows("elite"), "| best", rows("best"), "| sets", rows("sets-table"),
    "| charts", ["mo-chart", "cal-chart", "scores-chart"].map(id => !!d.querySelector(`#${id} svg`)).join(","));
  for (const b of d.querySelectorAll("#b-years button")) {
    b.click();
    console.log(`bracket ${b.dataset.y}: ${d.getElementById("b-year").textContent} | ${d.getElementById("b-picks").textContent}`);
  }
  d.querySelectorAll("#presets button")[2].click();
  console.log("preset:", d.getElementById("w-a-name").textContent, d.getElementById("w-a").textContent, "vs", d.getElementById("w-b-name").textContent, d.getElementById("w-b").textContent, "| params for", d.getElementById("pm-team").textContent);
  const bs = d.getElementById("b-season"); bs.value = "2022"; bs.dispatchEvent(new w.Event("change"));
  console.log("Team B season -> 2022:", d.getElementById("w-b-name").textContent);
  console.log("errors:", errs); process.exit(0);
}, 1500);
