// Maiman Studio, desktop edition.
//
// One window, showing the same studio page `maiman serve` serves to a browser,
// from a session server this process starts and owns. Everything the page
// does -- running a link, sweeping, saving and opening project files -- goes
// through that server exactly as it does in a browser tab, so the two can
// never disagree about a number.

const path = require("node:path");
const { app, BrowserWindow, Menu, dialog, shell } = require("electron");
const backend = require("./backend");

let server = null;
let window = null;

// Two copies would each start a server and each own a window; the second
// launch focuses the first instead.
if (!app.requestSingleInstanceLock()) {
  app.quit();
} else {
  app.on("second-instance", () => {
    if (!window) return;
    if (window.isMinimized()) window.restore();
    window.focus();
  });
  app.whenReady().then(launch);
}

async function launch() {
  Menu.setApplicationMenu(buildMenu());

  try {
    server = await backend.start({
      packaged: app.isPackaged,
      resourcesPath: process.resourcesPath,
      repoRoot: path.resolve(__dirname, ".."),
    });
  } catch (error) {
    const detail = [error.message, error.log].filter(Boolean).join("\n\n").slice(-4000);
    dialog.showErrorBox("Maiman Studio could not start its session server", detail);
    app.exit(1);
    return;
  }

  server.child.on("exit", (code, signal) => {
    if (app.isQuitting) return;
    dialog.showErrorBox(
      "The session server stopped",
      `It exited (${signal || `code ${code}`}). The studio cannot run anything without it, ` +
        `so the application will close.\n\n${server.log.join("").slice(-4000)}`,
    );
    app.exit(1);
  });

  createWindow(server.url);
}

function createWindow(url) {
  window = new BrowserWindow({
    width: 1440,
    height: 900,
    minWidth: 960,
    minHeight: 600,
    title: "Maiman Studio",
    show: false,
    webPreferences: {
      contextIsolation: true,
      nodeIntegration: false,
      sandbox: true,
    },
  });

  // The page is the studio and nothing else. A link that leads anywhere other
  // than the session server opens in the person's own browser, where it
  // belongs, rather than replacing the application inside its own window.
  const origin = new URL(url).origin;
  window.webContents.setWindowOpenHandler(({ url: target }) => {
    if (target.startsWith("http:") || target.startsWith("https:")) {
      shell.openExternal(target);
    }
    return { action: "deny" };
  });
  window.webContents.on("will-navigate", (event, target) => {
    if (new URL(target).origin !== origin) {
      event.preventDefault();
      shell.openExternal(target);
    }
  });

  if (process.platform !== "darwin") window.setMenuBarVisibility(false);
  window.once("ready-to-show", () => window.show());
  window.on("closed", () => {
    window = null;
  });
  window.loadURL(url);
}

function buildMenu() {
  const mac = process.platform === "darwin";
  // The studio draws its own menu bar (File, Edit, Simulate, View, Help) and
  // owns its shortcuts, so this menu stays out of the way. Accelerators here
  // are taken before the page sees the key, which is why there are no Undo,
  // Redo or zoom items: Cmd/Ctrl+Z and Cmd/Ctrl+= -/0 are the studio's undo
  // and canvas zoom, and a menu that claimed them would break both.
  //
  // On macOS a menu bar is not optional, and without cut/copy/paste roles in
  // it Cmd+C and Cmd+V do nothing in any text field. Elsewhere the bar is
  // hidden (see createWindow) and only its shortcuts remain.
  const help = {
    role: "help",
    submenu: [
      {
        label: "Maiman Studio on GitHub",
        click: () => shell.openExternal("https://github.com/ehsun-sh/maiman-studio"),
      },
      {
        label: "Report a problem",
        click: () =>
          shell.openExternal("https://github.com/ehsun-sh/maiman-studio/issues/new/choose"),
      },
    ],
  };
  return Menu.buildFromTemplate([
    ...(mac ? [{ role: "appMenu" }] : []),
    {
      label: "Edit",
      submenu: [{ role: "cut" }, { role: "copy" }, { role: "paste" }, { role: "selectAll" }],
    },
    {
      label: "View",
      submenu: [{ role: "reload" }, { role: "toggleDevTools" }, { role: "togglefullscreen" }],
    },
    ...(mac ? [{ role: "windowMenu" }] : []),
    help,
  ]);
}

app.on("before-quit", () => {
  app.isQuitting = true;
  if (server) server.stop();
});

app.on("window-all-closed", () => {
  // The server belongs to the window. On macOS the convention is that an app
  // outlives its last window, but a studio with no window has nothing to
  // show and a server nobody can reach, so it quits everywhere.
  app.quit();
});

// Belt and braces: whatever ends this process, the server goes with it.
process.on("exit", () => {
  if (server) server.stop();
});
