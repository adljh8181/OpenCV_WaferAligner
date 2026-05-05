# -*- mode: python ; coding: utf-8 -*-
# PyInstaller spec for WaferAligner UI
# Build with:  pyinstaller WaferAligner_UI.spec

import os

# Bundle any icon / logo files present in the workspace root
root_datas = []
for fname in ('QES.ico', 'logo.png'):
    if os.path.exists(fname):
        root_datas.append((fname, '.'))

# Bundle the recipes folder so default recipes ship with the exe
recipe_datas = []
if os.path.isdir('recipes'):
    recipe_datas.append(('recipes', 'recipes'))

# scipy has been eliminated — find_peaks and gaussian_filter1d are now
# implemented in pure numpy inside app/services/fov_classifier.py

a = Analysis(
    ['WaferAligner_UI.py'],
    pathex=['.'],
    binaries=[],
    datas=[
        ('app', 'app'),          # entire app package
        ('Images', 'Images'),    # FOV classifier reference images
        *root_datas,
        *recipe_datas,
    ],
    hiddenimports=[
        # tkinter and sub-modules
        'tkinter',
        'tkinter.ttk',
        'tkinter.filedialog',
        'tkinter.messagebox',
        # matplotlib Tk backend only
        'matplotlib.backends.backend_tkagg',
        'matplotlib.backends._backend_tk',
        # pystray platform backend (Windows)
        'pystray._win32',
        # pyzmq transports
        'zmq.backend.cython',
        'zmq.backend.cffi',
        # PIL / Pillow
        'PIL._tkinter_finder',
        # app internals (ensure they are found)
        'app.views.main_window',
        'app.views.pattern_tab',
        'app.views.edge_tab',
        'app.views.zmq_tab',
        'app.views.mask_editor',
        'app.viewmodels.pattern_viewmodel',
        'app.viewmodels.edge_viewmodel',
        'app.services.linemod_matcher',
        'app.services.edge_finder',
        'app.services.fov_classifier',
        'app.services.zmq_server',
        'app.models.app_state',
        'app.models.recipe_model',
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        # scipy is no longer used — eliminates ~150 MB of compiled DLLs
        'scipy',
        # Heavy ML / data-science packages not used at runtime
        'torch', 'torchvision', 'torchaudio', 'onnxruntime',
        'langchain', 'chromadb', 'huggingface_hub',
        'pandas', 'sklearn', 'skimage', 'statsmodels',
        'IPython', 'jupyter', 'notebook', 'ipykernel', 'ipywidgets',
        # Unused matplotlib backends (only TkAgg is used)
        'matplotlib.backends.backend_qt5agg',
        'matplotlib.backends.backend_qt5',
        'matplotlib.backends.backend_qtagg',
        'matplotlib.backends.backend_pdf',
        'matplotlib.backends.backend_svg',
        'matplotlib.backends.backend_ps',
        'matplotlib.backends.backend_wx',
        'matplotlib.backends.backend_wxagg',
        'matplotlib.backends.backend_cairo',
        'matplotlib.backends.backend_webagg',
        'matplotlib.backends.backend_nbagg',
        'matplotlib.backends.backend_gtk3agg',
        'matplotlib.backends.backend_gtk4agg',
    ],
    noarchive=False,
    optimize=1,
)

pyz = PYZ(a.pure)

# onedir mode: EXE is just a small launcher (~1 MB).
# All DLLs and data sit in the dist\WaferAlignerUI\ folder beside it.
# Zip that folder to distribute.
exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,   # binaries go into COLLECT, not embedded in EXE
    name='WaferAlignerUI',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=['vcruntime*.dll', 'msvcp*.dll', 'python*.dll'],
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    icon='QES.ico' if os.path.exists('QES.ico') else None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=['vcruntime*.dll', 'msvcp*.dll', 'python*.dll'],
    name='WaferAlignerUI',
)
