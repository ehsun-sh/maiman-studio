# Maiman Studio — desktop

The studio as a desktop application. It is the same page `maiman serve` gives a browser, in its
own window, talking to a session server the application starts for itself and stops when it
closes. Nothing in the engine or the page knows the difference: the desktop build is one more
client of the server, not a second interface to keep in step with the first.

## How it fits together

```
Electron (main.js)
  ├─ starts the session server  ── backend.js: `maiman serve --host 127.0.0.1 --port 0`
  │                                 and waits for the URL line it prints
  └─ opens a window on that URL ── the studio page, served by the server, unchanged
```

- **Port 0.** The operating system picks a free port, so the application never collides with a
  `maiman serve` already running in a terminal, or with a second copy of itself.
- **Loopback only.** The server binds `127.0.0.1`, exactly as `maiman serve` does by default.
- **One process, owned.** The server is the application's child: closing the window or quitting
  stops it. A second launch focuses the window that is already open.
- **The backend is frozen.** Installers carry the server built by PyInstaller
  ([maiman-backend.spec](maiman-backend.spec)), Python and NumPy included, so the person installing
  needs neither.
- **The page keeps its own menus and shortcuts.** On Windows and Linux the native menu bar is
  hidden, because the studio draws its own. On macOS the native menu carries only what macOS needs
  (cut, copy, paste, quit) and claims none of the studio's shortcuts: Cmd+Z and Cmd+= − 0 are the
  canvas's undo and zoom.

## Running it from a checkout

Needs Node 20+ and a Python that can import the engine.

```bash
cd desktop
npm install
npm start
```

`npm start` runs the server from `../src` with `python3` (`python` on Windows); set
`MAIMAN_PYTHON` to use another interpreter, or `MAIMAN_BACKEND` to point at any executable that
behaves like `maiman serve`.

## Building an installer

PyInstaller cannot cross-compile, so each platform is built on that platform.

```bash
pip install .. pyinstaller        # the engine, and the tool that freezes it
npm run backend                   # → build/backend/maiman-backend/
node smoke.js build/backend/maiman-backend/maiman-backend   # serves the page, runs a link
npm run dist                      # → dist/
```

| Platform | Output |
| :--- | :--- |
| Windows | `maiman-studio-<version>-win-x64.exe` (installer) and a `.zip` |
| macOS | `maiman-studio-<version>-mac-<arch>.dmg` and a `.zip` |
| Linux | `maiman-studio-<version>-linux-x86_64.AppImage` and a `.deb` |

The [Desktop workflow](../.github/workflows/desktop.yml) does all of this on Windows, macOS and
Linux runners, on any pull request that touches the desktop build, on demand, and on every
published release — where it attaches the installers to the release.

`version` in `package.json` must match the engine's; `tests/test_packaging.py` checks it.

## Opening an unsigned build

The installers are not code-signed yet, so each operating system warns once:

- **Windows** — SmartScreen says it protected your PC. Click **More info**, then **Run anyway**.
- **macOS** — "cannot be opened because the developer cannot be verified". Right-click the app in
  Applications, choose **Open**, and confirm. On recent macOS versions, open **System Settings →
  Privacy & Security** and click **Open Anyway**.
- **Linux** — make the AppImage executable (`chmod +x maiman-studio-*.AppImage`) and run it.

Signing needs an Apple Developer ID and a Windows code-signing certificate; electron-builder picks
both up from `CSC_LINK` / `CSC_KEY_PASSWORD` (and `APPLE_ID` etc. for notarisation) once they
exist.
