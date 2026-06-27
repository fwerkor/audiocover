# -*- mode: python ; coding: utf-8 -*-

import sys
from pathlib import Path

from PyInstaller.utils.hooks import collect_all, collect_data_files, collect_dynamic_libs

ROOT = Path(SPECPATH).resolve().parent
BUNDLE_ASSETS_DIR = ROOT / 'build' / 'audiocover-bundle-assets'

binaries = []
datas = [
    (str(ROOT / 'configs'), 'configs'),
    (str(ROOT / 'profiles' / 'example.yaml'), 'profiles'),
    (str(ROOT / 'README.md'), '.'),
    (str(ROOT / 'NOTICE.md'), '.'),
    (str(ROOT / 'LICENSE'), '.'),
]
hiddenimports = [
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
    'audiocover.workers.json_worker',
    'audiocover.workers.simple_timbre_worker',
    'audiocover.workers.demucs_separator_worker',
    'audiocover.workers.so_vits_svc_worker',
    'demucs.separate',
    'librosa',
    'pyloudnorm',
    'soundfile',
    'scipy.signal',
    'sklearn',
    'tensorboard',
    'torch',
    'torch.utils.tensorboard',
    'torch.utils.tensorboard.writer',
    'transformers.models.hubert.modeling_hubert',
]

if BUNDLE_ASSETS_DIR.exists():
    datas.append((str(BUNDLE_ASSETS_DIR), 'assets'))

for package in (
    'demucs',
    'librosa',
    'sklearn',
    'so_vits_svc_fork',
    'tensorboard',
    'torch',
    'torchaudio',
    'transformers',
):
    try:
        package_datas, package_binaries, package_hidden = collect_all(package)
    except Exception:
        package_datas, package_binaries, package_hidden = [], [], []
    datas += package_datas
    binaries += package_binaries
    hiddenimports += package_hidden

for package in ('soundfile', 'numpy', 'scipy'):
    try:
        datas += collect_data_files(package)
        binaries += collect_dynamic_libs(package)
    except Exception:
        pass

block_cipher = None

a = Analysis(
    [str(ROOT / 'src' / 'audiocover' / 'gui.py')],
    pathex=[str(ROOT), str(ROOT / 'src')],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        'IPython',
        'notebook',
        'pandas',
        'pyarrow',
        'pytest',
        'pycparser.lextab',
        'pycparser.yacctab',
        'scipy.special._cdflib',
        'torch.distributed._shard.checkpoint',
        'torch.distributed._sharded_tensor',
        'torch.distributed._sharding_spec',
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
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
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
