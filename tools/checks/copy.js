// Dump the rendered copy of every view to Markdown, so it can be read, linted and rewritten.
// usage: node copy.js ../../web/index.html <out-dir>
// writes <out-dir>/<view>.md (headings as #, text blocks as paragraphs) and <out-dir>/all.md
const fs = require("fs"), path = require("path");
const { JSDOM, VirtualConsole } = require("jsdom");
const [src, out = "copy-out"] = process.argv.slice(2);
if (!src) { console.error("usage: node copy.js <page.html> [out-dir]"); process.exit(1); }
const vc = new VirtualConsole(); vc.on("jsdomError", e => console.log("ERROR:", e.message));
const dom = new JSDOM(fs.readFileSync(src, "utf8"), { runScripts: "dangerously", pretendToBeVisual: true, virtualConsole: vc, url: "file:///page/index.html" });
const w = dom.window, d = w.document; w.scrollTo = () => {};
const SKIP = "script,style,svg,noscript,select,option,input,[hidden],[aria-hidden=true]";
const BLOCK = "h1,h2,h3,h4,p,li,figcaption,summary,caption,th,td,button,a,label,dt,dd,.lede,.verdict,.eyebrow,[role=listitem],.note";
const squash = s => s.replace(/\s+/g, " ").replace(/\s+([.,;:%)])/g, "$1").replace(/([(])\s+/g, "$1").trim();
// text of an element, with a space between child elements so "Round 1" and "beat X" don't run together
function textOf(el) {
  const parts = [];
  (function walk(n) {
    for (const c of n.childNodes) {
      if (c.nodeType === 3) parts.push(c.nodeValue);
      else if (c.nodeType === 1 && !c.matches(SKIP)) { parts.push(" "); walk(c); parts.push(" "); }
    }
  })(el);
  return squash(parts.join(""));
}

function dump(root) {
  const lines = [], seen = new Set();
  for (const el of root.querySelectorAll(BLOCK)) {
    const skipped = el.closest(SKIP);
    if (skipped && root.contains(skipped) && skipped !== root) continue;
    if (el.parentElement && el.parentElement.closest(BLOCK) && !/^H\d$/.test(el.tagName)) continue; // inner of a block already taken
    const t = textOf(el);
    if (!t || seen.has(t) || /^[\d\s.,%:+−-]+$/.test(t)) continue; // skip bare numbers and repeats
    seen.add(t);
    const h = /^H(\d)$/.exec(el.tagName);
    lines.push(h ? `${"#".repeat(+h[1])} ${t}` : t);
  }
  return lines.join("\n\n") + "\n";
}

setTimeout(() => {
  fs.mkdirSync(out, { recursive: true });
  const all = [];
  for (const v of ["home", "bracket", "matchups", "findings", "model"]) {
    w.location.hash = v; w.dispatchEvent(new w.HashChangeEvent("hashchange"));
    const view = d.querySelector(`.view[data-view="${v}"]`);
    if (!view) { console.log(`no view ${v}`); continue; }
    const md = dump(view);
    fs.writeFileSync(path.join(out, `${v}.md`), md);
    all.push(md);
    console.log(`${v}: ${md.split(/\s+/).length} words`);
  }
  // nav, footer and the chat panel (hidden until opened), plus the labels a screen reader or placeholder shows
  const chrome = [...d.querySelectorAll("header.top, footer, #chat")].map(dump).join("\n")
    + "\n" + [...new Set([...d.querySelectorAll("[placeholder], [aria-label]")].filter(e => !e.closest(".view"))
      .map(e => e.getAttribute("placeholder") || e.getAttribute("aria-label")))].map(t => `Label: ${t}`).join("\n\n") + "\n";
  fs.writeFileSync(path.join(out, "chrome.md"), chrome);
  fs.writeFileSync(path.join(out, "all.md"), all.join("\n") + `\n${chrome}`);
  console.log(`wrote ${out}/`); process.exit(0);
}, 1500);
