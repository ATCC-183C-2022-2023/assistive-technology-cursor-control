# -*- mode: python ; coding: utf-8 -*-


a = Analysis(
    ['mouse_and_keyboard_control_topkg.py'],
    pathex=[],
    binaries=[],
    datas=[('data/template_image_new_wide_data/', 'data/template_image_new_wide_data/'), ('mnist_model/saved_models/', 'mnist_model/saved_models/'), ('emnist_model/saved_models/', 'emnist_model/saved_models/'), ('/Library/Frameworks/Python.framework/Versions/3.11/lib/python3.11/site-packages/mediapipe', 'mediapipe')],
    hiddenimports=[],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='mouse_and_keyboard_control_topkg',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='mouse_and_keyboard_control_topkg',
)
