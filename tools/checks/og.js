// Social preview image: the home page hero at 1200x630, dark theme.
const puppeteer = require("puppeteer-core");
const ROOT = require("path").resolve(__dirname, "../.."), ROOT_URL = "file://" + ROOT + "/web/index.html";
const CHROME = process.env.CHROME || "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome";
(async () => {
  const b = await puppeteer.launch({ executablePath: CHROME, headless: "new" });
  const p = await b.newPage();
  await p.setViewport({ width: 1200, height: 630, deviceScaleFactor: 1 });
  await p.emulateMediaFeatures([{ name: "prefers-color-scheme", value: "dark" }]);
  await p.goto(ROOT_URL + "#home", { waitUntil: "load" });
  await new Promise(r => setTimeout(r, 800));
  // Hide the nav so the card reads as a title card.
  await p.addStyleTag({ content: ".top nav{display:none} .hero{padding-block:36px 24px}" });
  await p.screenshot({ path: ROOT + "/web/og.png", clip: { x: 0, y: 0, width: 1200, height: 630 } });
  await b.close();
  console.log("wrote web/og.png");
})();
