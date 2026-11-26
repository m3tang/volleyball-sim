// Runs the page's in-browser agent tools and grounding check (web/template.html, MATH and AGENT blocks)
// on the calls in argv[2] (JSON), printing results as JSON. Used by tests/test_site_agent.py.
const fs = require("fs"), path = require("path"), vm = require("vm");
const root = path.join(__dirname, "..");
const html = fs.readFileSync(path.join(root, "web/template.html"), "utf8");
const grab = (a, b) => html.slice(html.indexOf(a), html.indexOf(b) + b.length);
const code = grab("// MATH-START", "// MATH-END") + "\n" + grab("// AGENT-START", "// AGENT-END")
  + "\n;globalThis.__api = { agentTools, groundCheck, ToolError, AGENT_TOOL_SPECS };";
const ctx = vm.createContext({ console });
vm.runInContext(code, ctx);
const { agentTools, groundCheck, ToolError } = ctx.__api;
const D = JSON.parse(fs.readFileSync(path.join(root, "web/data.json"), "utf8"));
const tools = agentTools(D);
const jobs = JSON.parse(fs.readFileSync(process.argv[2], "utf8"));
const out = jobs.map(j => {
  if (j.ground) return { ground: groundCheck(j.ground.answer, j.ground.calls, j.ground.question || "") };
  try { return { result: tools[j.name](j.args) }; }
  catch (e) { return { error: e instanceof ToolError ? e.message : "CRASH " + e.stack }; }
});
process.stdout.write(JSON.stringify(out));
