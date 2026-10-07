"""Open Maiman Studio in the browser, from this checkout, with nothing but Python.

This is what the "Maiman App" shortcut runs (see tools/make_shortcut.py). It
starts the session server on a free loopback port, opens the studio in the
default browser, and shows a small window that keeps the server alive: closing
that window stops Maiman. Without Tk (some Linux Pythons ship without it) the
server runs until Ctrl+C instead.
"""

from __future__ import annotations

import contextlib
import sys
import webbrowser
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))


def _tk():
    try:
        import tkinter

        return tkinter
    except ImportError:
        return None


def main() -> int:
    tk = _tk()
    try:
        from maiman import server
    except ImportError as exc:
        message = (
            f"Maiman needs Python 3.11+ with NumPy.\n\n{exc}\n\n"
            f'Run:  {Path(sys.executable).name} -m pip install -e "{ROOT}"'
        )
        if tk is not None:
            root = tk.Tk()
            root.withdraw()
            from tkinter import messagebox

            messagebox.showerror("Maiman App", message)
        else:
            print(message, file=sys.stderr)
        return 1

    httpd = server.serve("127.0.0.1", 0)
    url = f"http://127.0.0.1:{httpd.server_address[1]}/"
    webbrowser.open(url)

    try:
        if tk is None:
            print(f"Maiman Studio is running at {url}\nPress Ctrl+C to stop.")
            import threading

            threading.Event().wait()
        else:
            root = tk.Tk()
            root.title("Maiman App")
            root.resizable(False, False)
            icon = ROOT / "assets" / "icon-192.png"
            with contextlib.suppress(tk.TclError):
                root.iconphoto(True, tk.PhotoImage(file=str(icon)))
            tk.Label(root, text="Maiman Studio is running", font=("", 12, "bold")).pack(
                padx=24, pady=(16, 4)
            )
            tk.Label(root, text=url).pack(padx=24)
            buttons = tk.Frame(root)
            buttons.pack(padx=24, pady=16)
            tk.Button(buttons, text="Open studio", command=lambda: webbrowser.open(url)).pack(
                side="left", padx=4
            )
            tk.Button(buttons, text="Quit", command=root.destroy).pack(side="left", padx=4)
            root.mainloop()
    except KeyboardInterrupt:
        pass
    finally:
        httpd.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
