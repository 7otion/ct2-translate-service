import subprocess
import sys
import os
from pathlib import Path

# Configuration
SCRIPT = "ct2_translate_service.py"
DIST_DIR = "dist"
#TRANSLATION_MODELS = "translation_models"

PYINSTALLER_ARGS = [
    "--onedir",
    #f"--add-data={TRANSLATION_MODELS};{TRANSLATION_MODELS}",
    "--collect-all", "ctranslate2",
    "--collect-all", "sentencepiece",
    "--exclude-module", "tkinter",
    "--exclude-module", "matplotlib",
    "--exclude-module", "PIL",
    "--exclude-module", "cv2",
    SCRIPT
]

def build():
    """Run PyInstaller to build the executable."""
    # Clean previous build/dist if present
    for folder in ["build", DIST_DIR]:
        if Path(folder).exists():
            print(f"Removing {folder}...")
            if Path(folder).is_dir():
                import shutil
                shutil.rmtree(folder)
            else:
                Path(folder).unlink()

    # Run PyInstaller
    cmd = [sys.executable, "-m", "PyInstaller"] + PYINSTALLER_ARGS
    print("Running:", " ".join(cmd))
    result = subprocess.run(cmd)
    if result.returncode != 0:
        print("Build failed.")
        sys.exit(result.returncode)
    print(f"Build complete. Executable is in {DIST_DIR}/")

if __name__ == "__main__":
    build()
