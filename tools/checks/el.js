// usage: node el.js <view> <css selector> <out.png> [width]  -> one screenshot per matching element
const puppeteer = require("puppeteer-core");
const ROOT = require("path").resolve(__dirname, "../.."), ROOT_URL = "file://" + ROOT + "/web/index.html";
const CHROME = process.env.CHROME || "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome";
(async () => {
  const [view, sel, out, width] = process.argv.slice(2);
  const b = await puppeteer.launch({ executablePath: CHROME, headless: "new" });
  const p = await b.newPage();
  const errs = [];
  p.on("pageerror", e => errs.push(e.message));
  await p.setViewport({ width: +(width || 1280), height: 900 });
  await p.emulateMediaFeatures([{ name: "prefers-color-scheme", value: "dark" }]);
  await p.goto(ROOT_URL + "#" + view, { waitUntil: "load" });
  await new Promise(r => setTimeout(r, 600));
  const els = await p.$$(sel);
  for (let i = 0; i < els.length; i++) await els[i].screenshot({ path: out.replace(".png", "-" + i + ".png") });
  console.log(sel, els.length, "errors:", errs.length ? errs : 0);
  await b.close();
})();
