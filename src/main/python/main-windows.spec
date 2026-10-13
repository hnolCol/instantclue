# -*- mode: python ; coding: utf-8 -*-
import os, shutil, sys, importlib.util
from PyInstaller.utils.hooks import collect_all, collect_submodules, collect_data_files
block_cipher = None

# scikit-learn's KNN/pairwise-distance backend (used by NearestNeighbors, and
# transitively by UMAP/openTSNE) is split across several compiled Cython
# submodules under sklearn.metrics._pairwise_distances_reduction (added in
# scikit-learn 1.2.0, pinned here as 1.2.1). PyInstaller's own bundled
# pyinstaller-hooks-contrib hook for this only exists in hooks-contrib
# releases published after scikit-learn 1.2.0 shipped; the version pinned in
# requirements.txt (pyinstaller-hooks-contrib==2022.10) predates it entirely
# and adds no hidden imports for these submodules at all. The result:
# `_base.pyx`'s init code does a C-level import of `_datasets_pair` that
# PyInstaller's bytecode scanner cannot see (same class of problem as
# openTSNE._matrix_mul, item 26), and the frozen app crashes at runtime with
# "ModuleNotFoundError: No module named
# 'sklearn.metrics._pairwise_distances_reduction._datasets_pair'" the moment
# any KNN/pairwise-distance code path runs.
# Fix (item 28): collect all of that subpackage's submodules explicitly via
# collect_submodules(), independent of whatever hooks-contrib version is
# installed. Verified by reproduction: built a frozen binary using
# NearestNeighbors.kneighbors() with the exact pinned pyinstaller==5.12.0 /
# pyinstaller-hooks-contrib==2022.10 combination - it reproduced the user's
# exact error message without this hiddenimport, and ran correctly with it.
_sklearn_pdr_hiddenimports = collect_submodules('sklearn.metrics._pairwise_distances_reduction')

# wordcloud/wordcloud.py reads a plain-text 'stopwords' data file (and
# references a bundled DroidSansMono.ttf font) at MODULE IMPORT time via a
# path relative to the package directory - these are not Python modules, so
# PyInstaller's bytecode-scanning Analysis never discovers them on its own.
# pyinstaller-hooks-contrib does have a hook for this (hook-wordcloud.py,
# added 2023) that does exactly the collect_data_files() call below - but the
# pinned pyinstaller-hooks-contrib==2022.10 predates that hook entirely (same
# class of gap as items 26/28). Crash without this fix:
#   FileNotFoundError: [Errno 2] No such file or directory:
#   '...\\_MEI.......\\wordcloud\\stopwords'
# Verified by reproduction: a frozen build with the exact pinned
# pyinstaller-hooks-contrib==2022.10 reproduced this exact error importing
# wordcloud (used by backend/plotting/plotterCalculations.py); adding
# collect_data_files('wordcloud') to datas resolved it.
_wordcloud_datas = collect_data_files('wordcloud')

# openTSNE._matrix_mul is a compiled Cython extension that a *different*
# compiled module (openTSNE._tsne) reaches via a C-level PyImport_ImportModule
# call baked into its init code - not a Python-level import statement
# PyInstaller's bytecode scanner can see, and not a linked shared library its
# binary-dependency walker can see either. Declaring it as a plain string in
# hiddenimports (as this spec previously did) does NOT work - PyInstaller's
# hidden-import resolver independently fails to locate it even though it is a
# perfectly normal, importable package (confirmed: "Hidden import
# 'openTSNE._matrix_mul' not found" during Analysis, yet `import
# openTSNE._matrix_mul.matrix_mul` succeeds in a plain, non-frozen Python).
#
# First attempt used collect_all('openTSNE') (see CHANGELOG.md item 25),
# verified to work in a from-scratch reproduction here. It failed on the
# user's actual build machine with:
#   "Unable to copy metadata for openTSNE: The 'openTSNE' distribution was
#   not found and is required by the application"
#   "collect_data_files/collect_dynamic_libs - skipping ... as it is not a
#   package"
# collect_all()'s package-detection path goes through pkg_resources-based
# distribution metadata lookup, which depends on how openTSNE happened to get
# installed in that specific venv. Notably, openTSNE is not even listed in
# requirements.txt - it's an untracked dependency InstantClue's own code
# imports directly (backend/statistics/dimensionalReduction/ICPCA.py,
# backend/statistics/statistics.py), so whatever version and install method
# ended up in that venv was never pinned or verified, and evidently didn't
# register metadata the way a plain `pip install openTSNE` does.
#
# Fix (item 26): collect the package's files directly via
# importlib.util.find_spec() - the exact same lookup mechanism a plain
# `import openTSNE` statement uses at runtime, with no dependency on
# pkg_resources/dist-info metadata at all. Since the frozen app crashed
# *inside* openTSNE's own compiled code (proving `import openTSNE` already
# works fine in that venv), find_spec() is guaranteed to succeed there too.
# Verified end-to-end here: built a minimal frozen binary using only this
# find_spec-based walk (no collect_all involved) and confirmed it runs
# openTSNE successfully, with none of the "distribution not found"/"not a
# package" warnings collect_all triggered.
def _collect_package_files(package_name):
    spec = importlib.util.find_spec(package_name)
    if spec is None or spec.submodule_search_locations is None:
        raise RuntimeError(
            f"{package_name} not found as a package via find_spec() in this "
            f"interpreter/venv ({sys.executable}). This means it is genuinely "
            f"not installed here (not a metadata-registration issue) - run "
            f"`pip show {package_name}` to confirm, then "
            f"`pip install {package_name}` into THIS venv before re-running "
            f"pyinstaller. Note: openTSNE is not listed in requirements.txt, "
            f"so a venv built purely from that file will be missing it.")
    pkg_dir = list(spec.submodule_search_locations)[0]
    site_root = os.path.dirname(pkg_dir)
    datas = []
    for root, dirs, files in os.walk(pkg_dir):
        dirs[:] = [d for d in dirs if d != '__pycache__']
        for fn in files:
            if fn.endswith('.pyc'):
                continue
            full = os.path.join(root, fn)
            dest_dir = os.path.relpath(root, site_root)
            datas.append((full, dest_dir))
    return datas

# pynndescent is included via collect_all as before - its metadata lookup
# (importlib.metadata.version("pynndescent"), unprotected by a try/except in
# its own __init__.py) resolved cleanly via collect_all on the user's build
# (no "distribution not found" warning was logged for it), so only openTSNE
# needed the more defensive find_spec-based approach.
_extra_datas, _extra_binaries, _extra_hiddenimports = [], [], []
_extra_datas += _collect_package_files('openTSNE')
for _pkg in ('pynndescent',):
    _d, _b, _h = collect_all(_pkg)
    _extra_datas += _d
    _extra_binaries += _b
    _extra_hiddenimports += _h

# SPECPATH is injected by PyInstaller into this file's exec namespace and
# points at the directory containing this .spec file (src/main/python).
# Using it instead of a hardcoded developer path makes this spec portable
# to any machine/clone location.
BASE_DIR = SPECPATH

def _copytree_safe(src, dst):
    if os.path.exists(dst):
        shutil.rmtree(dst)
    shutil.copytree(src, dst)

# Item 65: conf/ holds per-user runtime data only (saved settings current.ic,
# conf/key/app_id, conf/mainFigureImages) and is excluded from the upstream git
# repository (.gitignore), so it is not part of a fresh source tree. InstantClue
# creates it itself on first start, but this spec copies and bundles conf/ and
# conf/key/ below, so building a fresh tree before ever running "python main.py"
# failed with "FileNotFoundError: [WinError 3] ... src\\main\\python\\conf".
# Create the (empty) folders if they are missing; nothing is changed if they exist.
os.makedirs(os.path.join(BASE_DIR, 'conf', 'key'), exist_ok=True)

_copytree_safe(os.path.join(BASE_DIR, 'examples'), os.path.join(BASE_DIR, 'dist', 'examples'))
_copytree_safe(os.path.join(BASE_DIR, 'conf'), os.path.join(BASE_DIR, 'dist', 'conf'))
_copytree_safe(os.path.join(BASE_DIR, 'annotations'), os.path.join(BASE_DIR, 'dist', 'annotations'))
_copytree_safe(os.path.join(BASE_DIR, 'icons'), os.path.join(BASE_DIR, 'dist', 'icons'))
_copytree_safe(os.path.join(BASE_DIR, 'quickSelectLists'), os.path.join(BASE_DIR, 'dist', 'quickSelectLists'))


a = Analysis(['main.py'],
             pathex=[BASE_DIR],
             binaries=_extra_binaries,
             datas=_extra_datas + _wordcloud_datas,
             hooksconfig={
                "matplotlib": {"backends": "all"}},
             # Trimmed to entries that actually resolve against the pinned dependency
             # versions in requirements.txt (verified against the user's own PyInstaller
             # build log - see CHANGELOG.md item 24). The removed entries
             # ('sklearn.neighbors.typedefs'/'_typedefs'/'quad_tree', 'numpy.random.common'/
             # 'bounded_integers'/'entropy', 'openTSNE._matrix_mul'/'.matrix_mul') were leftovers
             # from a much older scikit-learn/numpy/openTSNE and no longer exist under those
             # names - PyInstaller logged them as "Hidden import not found" (non-fatal - the
             # Analysis step skips them and the build still completes) but they were dead
             # weight causing log noise on every build. openTSNE/pynndescent's real hidden
             # imports now come from collect_all() above (item 25), not a manual string list.
             hiddenimports=_extra_hiddenimports + _sklearn_pdr_hiddenimports + ['pynndescent','sklearn.utils.murmurhash','sklearn.tree._utils',
             				'sklearn.neighbors._partition_nodes','scipy._lib.messagestream',
                                   'scipy.special.cython_special','sklearn.utils._cython_blas','openpyxl'],
             hookspath=[],
             runtime_hooks=[],
             excludes=[],
             win_no_prefer_redirects=False,
             win_private_assemblies=False,
             cipher=block_cipher,
             noarchive=False)


splash = Splash(os.path.join(BASE_DIR, 'icons', 'splashScreen.jpg'),
                binaries=a.binaries,
                datas=a.datas,
                text_pos=(10, 50),
                text_size=12,
                text_color='black')



a.datas += Tree(os.path.join(BASE_DIR, 'examples'), prefix='examples')
a.datas += Tree(os.path.join(BASE_DIR, 'annotations'), prefix='annotations')
a.datas += Tree(os.path.join(BASE_DIR, 'conf'), prefix='conf')
a.datas += Tree(os.path.join(BASE_DIR, 'conf', 'key'), prefix='key')
a.datas += Tree(os.path.join(BASE_DIR, 'icons'), prefix='icons')
a.datas += Tree(os.path.join(BASE_DIR, 'quickSelectLists'), prefix='quickSelectLists')

pyz = PYZ(a.pure, a.zipped_data,
             cipher=block_cipher)

# Item 42 (slow startup): switched from ONEFILE to ONEDIR, and disabled UPX.
#
# The previous EXE(pyz, a.scripts, a.binaries, a.zipfiles, a.datas, splash,
# splash.binaries, [], ...) call with runtime_tmpdir=None was a ONEFILE build:
# every launch, the bootloader has to re-extract the *entire* ~190 MB payload
# (Python interpreter, Qt6, numpy/scipy/sklearn, numba's LLVM machinery,
# matplotlib, all bundled data) into a fresh temp directory before any app
# code runs - that extraction (plus, on many corporate machines, antivirus
# re-scanning those freshly-written temp files on every single launch) is
# almost certainly the dominant cost in "it takes long to load", not Python
# import time or app logic.
#
# ONEDIR avoids this entirely: binaries/data are unpacked once at build time
# into a stable output folder sitting next to the .exe, and every subsequent
# launch just runs directly against files already on disk - no self-extraction
# step. This is a pure win here because the project already distributes the
# whole dist\InstantClue\ folder zipped up (see CHANGELOG.md items 40/41), not
# a bare single .exe file, so there is no loss of "one portable file" - the
# folder was already the unit of distribution.
#
# UPX (upx=True) additionally compresses every bundled DLL/pyd, which adds
# decompression overhead each time Windows loads one of those libraries, and
# is a common source of false-positive antivirus flags on compressed
# binaries - another possible contributor to "slow to load" on locked-down
# machines. Disabled here (upx=False) to trade a larger folder-on-disk size
# for faster, more predictable loading. Re-enable if disk footprint matters
# more than launch speed.
#
# sys._MEIPASS (used by the item 31/40 fixes for self.mainPath/filePath) is
# set by PyInstaller in ONEDIR builds too - it just points directly at this
# already-unpacked folder instead of a fresh temp dir, so those fixes need no
# further changes.
#
# runtime_tmpdir was removed - it is a ONEFILE-only setting (controls where
# the self-extraction happens) and has no effect once exclude_binaries=True.
#
# splash (item 44 fix): PyInstaller 5.12's EXE() special-cases `Splash`
# instances as a valid positional argument (it records the splash resource
# and merges its dependency TOC) - see PyInstaller.building.api.EXE.__init__.
# COLLECT(), by contrast, only accepts EXE instances or TOC-like iterables;
# it has no special case for a bare Splash object and raises TypeError on
# one. So `splash` itself belongs on EXE, and only the iterable
# `splash.binaries` belongs on COLLECT (see below).
exe = EXE(pyz,
          a.scripts,
          [],
          splash,
          exclude_binaries=True,
          name='InstantClue',
          debug=False,
          bootloader_ignore_signals=False,
          strip=False,
          upx=False,
          upx_exclude=[],
          # Item 55: console=False (the item-29 temporary console=True is reverted).
          # A console window is not wanted in the shipped app. Errors are no
          # longer lost without a console: main.py now writes them to
          # InstantClue_error_log.txt (Python errors) and InstantClue_crash_log.txt
          # (hard crashes) in the installation folder, next to InstantClue.exe.
          console=False,
          icon = os.path.join(BASE_DIR, 'icons', 'ICLogo.ico'))

coll = COLLECT(exe,
                a.binaries,
                a.zipfiles,
                a.datas,
                splash.binaries,
                strip=False,
                upx=False,
                upx_exclude=[],
                name='InstantClue')
