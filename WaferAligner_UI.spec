# -*- mode: python ; coding: utf-8 -*-
# PyInstaller spec for WaferAligner UI
# Build with:  pyinstaller WaferAligner_UI.spec

import os
from PyInstaller.utils.hooks import collect_data_files, collect_dynamic_libs

# Bundle any icon / logo files present in the workspace root
root_datas = []
for fname in ('QES.ico', 'logo.png'):
    if os.path.exists(fname):
        root_datas.append((fname, '.'))

# Bundle the recipes folder so default recipes ship with the exe
recipe_datas = []
if os.path.isdir('recipes'):
    recipe_datas.append(('recipes', 'recipes'))

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
        # matplotlib Tk backend
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
        # Heavy ML packages not needed at runtime — reduces exe size significantly
        'torch', 'torchvision', 'onnxruntime',
        'langchain', 'chromadb', 'huggingface_hub',
        'pandas', 'scipy', 'sklearn',
        'IPython', 'jupyter', 'notebook',
    ],
    noarchive=False,
    optimize=1,
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name='WaferAlignerUI',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=['vcruntime*.dll', 'msvcp*.dll', 'python*.dll'],
    runtime_tmpdir=None,
    console=False,           # No black console window (GUI app)
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    icon='QES.ico' if os.path.exists('QES.ico') else None,
)
