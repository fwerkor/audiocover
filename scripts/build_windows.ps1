$ErrorActionPreference = "Stop"
python -m pip install -U pip
python -m pip install -e ".[build]"
pyinstaller packaging/audiocover-gui.spec --clean --noconfirm
Compress-Archive -Path dist/AudioCoverLab/* -DestinationPath dist/AudioCoverLab-windows.zip -Force
Write-Host "Built dist/AudioCoverLab-windows.zip"
