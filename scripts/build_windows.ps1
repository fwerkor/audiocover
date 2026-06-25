$ErrorActionPreference = "Stop"
python -m pip install -U pip
python -m pip install -e ".[build]"
pyinstaller packaging/audiocover-gui.spec --clean --noconfirm
Compress-Archive -Path dist/AudioCover/* -DestinationPath dist/AudioCover-windows.zip -Force
Write-Host "Built dist/AudioCover-windows.zip"
