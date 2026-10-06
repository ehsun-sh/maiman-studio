// The session server, started and stopped by the desktop shell.
//
// The browser version asks a person to run `maiman serve` and open the URL it
// prints. This does the same two things for them: it starts the server on a
// port the operating system chooses, waits for the line carrying the URL, and
// hands that URL to the window. The server is unchanged -- the desktop build
// is one more client of it, not a fork of it.

const { spawn } = require("node:child_process");
const fs = require("node:fs");
const path = require("node:path");

// The URL line `maiman serve` prints once the socket is listening.
const READY = /(http:\/\/127\.0\.0\.1:\d+\/)/;

// How long a first launch may take before it is reported as a failure. A
// frozen backend on a cold disk, with antivirus scanning every DLL it loads,
// can take tens of seconds the first time and well under one after.
const START_TIMEOUT_MS = 90_000;

/**
 * Decide what to run. In order:
 *   1. MAIMAN_BACKEND, a path to any executable that behaves like `maiman serve`.
 *   2. The frozen backend packaged with the application.
 *   3. MAIMAN_PYTHON (default `python3`, `python` on Windows) running
 *      `-m maiman.cli serve` -- what `npm start` uses from a checkout.
 */
function command({ packaged, resourcesPath, repoRoot }) {
  const args = ["--host", "127.0.0.1", "--port", "0"];

  if (process.env.MAIMAN_BACKEND) {
    return { file: process.env.MAIMAN_BACKEND, args };
  }

  const exe = process.platform === "win32" ? "maiman-backend.exe" : "maiman-backend";
  const bundled = path.join(resourcesPath, "backend", exe);
  if (packaged && fs.existsSync(bundled)) {
    return { file: bundled, args };
  }

  const python =
    process.env.MAIMAN_PYTHON || (process.platform === "win32" ? "python" : "python3");
  // From a checkout, put src/ on the path so `npm start` runs the code in the
  // working tree without needing `pip install -e .` first.
  const src = path.join(repoRoot, "src");
  const env = fs.existsSync(src)
    ? { PYTHONPATH: [src, process.env.PYTHONPATH].filter(Boolean).join(path.delimiter) }
    : {};
  return { file: python, args: ["-m", "maiman.cli", "serve", ...args], env };
}

/**
 * Start the server. Resolves with { url, stop, log } once it is listening;
 * rejects with an Error carrying the server's output if it exits or stalls
 * first.
 */
function start(options) {
  const { file, args, env } = command(options);
  const child = spawn(file, args, {
    env: { ...process.env, PYTHONUNBUFFERED: "1", ...env },
    stdio: ["ignore", "pipe", "pipe"],
    windowsHide: true,
  });

  // Kept for the error dialog. A server that fails to start says why on
  // stderr, and that sentence is worth more than any message written here.
  const log = [];
  const remember = (chunk) => {
    log.push(chunk.toString());
    if (log.length > 200) log.shift();
  };
  child.stderr.on("data", remember);

  let stopped = false;
  const stop = () => {
    if (stopped || child.exitCode !== null) return;
    stopped = true;
    child.kill();
  };

  return new Promise((resolve, reject) => {
    let buffered = "";
    const fail = (message) => {
      clearTimeout(timer);
      stop();
      const error = new Error(message);
      error.log = log.join("");
      reject(error);
    };
    const timer = setTimeout(
      () => fail(`the session server did not start within ${START_TIMEOUT_MS / 1000} s`),
      START_TIMEOUT_MS,
    );

    child.stdout.on("data", (chunk) => {
      remember(chunk);
      buffered += chunk.toString();
      const match = buffered.match(READY);
      if (match) {
        clearTimeout(timer);
        buffered = "";
        resolve({ url: match[1], stop, log, child });
      }
    });
    child.on("error", (error) => fail(`could not run ${file}: ${error.message}`));
    child.on("exit", (code, signal) => {
      if (!stopped) fail(`the session server exited (${signal || `code ${code}`})`);
    });
  });
}

module.exports = { start, command };
