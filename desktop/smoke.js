// Start the frozen backend the way the application does, and run a link
// through it. A frozen build can import cleanly and still be missing the
// studio page or a component that is only ever reached by name; this catches
// both before the backend is wrapped into an installer nobody can debug.
//
//   node smoke.js build/backend/maiman-backend/maiman-backend[.exe]

const fs = require("node:fs");
const path = require("node:path");
const backend = require("./backend");

async function main() {
  process.env.MAIMAN_BACKEND = path.resolve(process.argv[2]);
  const server = await backend.start({ packaged: false, resourcesPath: "", repoRoot: "" });
  try {
    const page = await fetch(server.url);
    const html = await page.text();
    if (page.status !== 200 || !html.includes("Maiman Studio")) {
      throw new Error(`the studio page came back ${page.status}`);
    }
    const project = fs.readFileSync(path.join(__dirname, "..", "examples", "maiman", "ook_eye.maiman"));
    const run = await fetch(new URL("api/run", server.url), { method: "POST", body: project });
    const body = await run.text();
    if (run.status !== 200) throw new Error(`a run came back ${run.status}: ${body.slice(0, 500)}`);
    console.log(`ok: ${server.url} served the studio and ran ook_eye.maiman`);
  } finally {
    server.stop();
  }
}

main().catch((error) => {
  console.error(error.message);
  if (error.log) console.error(error.log);
  process.exit(1);
});
