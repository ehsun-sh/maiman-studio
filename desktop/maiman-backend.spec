# PyInstaller recipe for the backend the desktop build bundles.
#
# One directory, not one file. A one-file build unpacks itself into a temporary
# directory on every launch and runs as two processes; on Windows, ending the
# outer one leaves the inner one -- the actual server -- running with nobody to
# stop it. One directory is one process, and it starts faster.
#
# collect_data_files is what carries the studio page and the PCS block order:
# both are package data, and an analysis that follows imports alone sees
# neither.

from PyInstaller.utils.hooks import collect_data_files, collect_submodules

a = Analysis(
    ["maiman-backend.py"],
    datas=collect_data_files("maiman"),
    # Components register themselves on import, and some are reached by name
    # from a project file rather than by an import PyInstaller can see.
    hiddenimports=collect_submodules("maiman"),
    excludes=["tkinter", "matplotlib", "pytest", "IPython"],
)
pyz = PYZ(a.pure)
exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="maiman-backend",
    console=True,
)
coll = COLLECT(exe, a.binaries, a.datas, name="maiman-backend")
