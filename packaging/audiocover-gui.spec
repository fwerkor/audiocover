# -*- mode: python ; coding: utf-8 -*-

import sys
from pathlib import Path

ROOT = Path(SPECPATH).resolve().parent
RUNTIME_DIR = ROOT / 'backend-runtimes'

datas = [
    (str(ROOT / 'configs'), 'configs'),
    (str(ROOT / 'profiles' / 'example.yaml'), 'profiles'),
    (str(ROOT / 'README.md'), '.'),
    (str(ROOT / 'NOTICE.md'), '.'),
    (str(ROOT / 'LICENSE'), '.'),
]
if RUNTIME_DIR.exists():
    datas.append((str(RUNTIME_DIR), 'backend-runtimes'))

block_cipher = None

a = Analysis(
    [str(ROOT / 'src' / 'audiocover' / 'gui.py')],
    pathex=[str(ROOT), str(ROOT / 'src')],
    binaries=[],
    datas=datas,
    hiddenimports=[
        'audiocover',
        'audiocover.cli',
        'audiocover.config',
        'audiocover.audio',
        'audiocover.dataset',
        'audiocover.training',
        'audiocover.pipeline',
        'audiocover.runtime',
        'audiocover.simple_timbre',
        'audiocover.qc',
        'audiocover.workers',
        'pyloudnorm',
        'soundfile',
        'scipy.signal',
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        'IPython',
        'librosa',
        'llvmlite',
        'matplotlib',
        'notebook',
        'numba',
        'pandas',
        'pyarrow',
        'pytest',
        'pycparser.lextab',
        'pycparser.yacctab',
        'scipy.special._cdflib',
        'sklearn',
        'torch',
        'torch._inductor',
        'torch.distributed._shard.checkpoint',
        'torch.distributed._sharded_tensor',
        'torch.distributed._sharding_spec',
        'torch.utils.tensorboard',
        'triton',
    ],
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
    name='AudioCover',
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
    name='AudioCover',
)

if sys.platform == 'darwin':
    app = BUNDLE(
        coll,
        name='AudioCover.app',
        icon=None,
        bundle_identifier='com.fwerkor.audiocover',
    )
