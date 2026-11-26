const puppeteer = require("puppeteer-core");
const ROOT = require("path").resolve(__dirname, "../.."), ROOT_URL = "file://" + ROOT + "/web/index.html";
const CHROME = process.env.CHROME || "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome";
(async () => {
  const b = await puppeteer.launch({ executablePath: CHROME, headless: "new" });
  const p = await b.newPage();
  await p.setViewport({ width: 1280, height: 860, deviceScaleFactor: 1.5 });
  await p.emulateMediaFeatures([{ name: "prefers-color-scheme", value: "dark" }]);
  const out = ROOT + "/docs/images/";
  await p.goto("https://m3tang.github.io/volleyball-sim/#home", { waitUntil: "networkidle0" });
  await p.screenshot({ path: out + "home.jpg", type: "jpeg", quality: 82 });
  await p.goto("https://m3tang.github.io/volleyball-sim/#matchups", { waitUntil: "networkidle0" });
  await p.evaluate(() => window.scrollTo(0, 0));
  await p.screenshot({ path: out + "matchups.jpg", type: "jpeg", quality: 82 });
  await b.close();
})();
