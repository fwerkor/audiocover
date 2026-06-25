# -*- mode: python ; coding: utf-8 -*-

block_cipher = None

a = Analysis(
    ['src/audiocover_lab/gui.py'],
    pathex=['.'],
    binaries=[],
    datas=[
        ('configs', 'configs'),
        ('profiles/example.yaml', 'profiles'),
        ('README.md', '.'),
        ('NOTICE.md', '.'),
        ('LICENSE', '.'),
    ],
    hiddenimports=[
        'audiocover_lab',
        'audiocover_lab.cli',
        'audiocover_lab.config',
        'audiocover_lab.audio',
        'audiocover_lab.dataset',
        'audiocover_lab.training',
        'audiocover_lab.pipeline',
        'audiocover_lab.simple_timbre',
        'audiocover_lab.qc',
        'librosa',
        'pyloudnorm',
        'soundfile',
        'scipy.signal',
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=['matplotlib', 'notebook', 'IPython'],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='AudioCoverLab',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='AudioCoverLab',
)
