// usage: node pp.js <view> <width> <out.png> [mobile]
const puppeteer = require("puppeteer-core");
const ROOT = require("path").resolve(__dirname, "../.."), ROOT_URL = "file://" + ROOT + "/web/index.html";
const CHROME = process.env.CHROME || "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome";
(async () => {
  const [view, width, out, mobile] = process.argv.slice(2);
  const browser = await puppeteer.launch({ executablePath: CHROME, headless: "new" });
  const page = await browser.newPage();
  page.on("pageerror", e => console.log("PAGE ERROR:", e.message));
  await page.setViewport({ width: +width, height: 900, isMobile: !!mobile, hasTouch: !!mobile, deviceScaleFactor: 1 });
  await page.emulateMediaFeatures([{ name: "prefers-color-scheme", value: "dark" }]);
  await page.goto(ROOT_URL + "#" + view, { waitUntil: "load" });
  await new Promise(r => setTimeout(r, 800));
  const info = await page.evaluate(() => {
    const vw = document.documentElement.clientWidth, wide = [];
    for (const el of document.querySelectorAll(".view:not([hidden]) *")) {
      const r = el.getBoundingClientRect();
      if (r.width && r.right > vw + 1 && !el.closest(".scroll, .formula")) wide.push(`${el.tagName.toLowerCase()}.${el.className || ""}#${el.id || ""} right=${Math.round(r.right)}`);
    }
    return { vw, scrollW: document.documentElement.scrollWidth, wide: wide.slice(0, 8) };
  });
  console.log(`${view} @${width}${mobile ? " mobile" : ""}: viewport ${info.vw}, page width ${info.scrollW}`, info.wide.length ? "\n  too wide: " + info.wide.join("\n  ") : "(no overflow)");
  await page.screenshot({ path: out, fullPage: true });
  await browser.close();
})();
