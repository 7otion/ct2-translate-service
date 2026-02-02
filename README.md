# CTranslate2 Translation Service

A minimal Python wrapper service for CTranslate2 for fast neural machine translation.

## Features

- Supports multiple language pairs (English, Spanish, French, German, Turkish, Arabic, Chinese, Russian, Portuguese, Italian, Japanese, Korean)
- Automatic model downloading from Argos repository
- Optimized for CPU with int8 quantization
- Handles OCR text normalization
- Post-processing to fix common translation issues

## Requirements

- Python 3.7+
- ctranslate2
- sentencepiece

Install dependencies:
```
pip install ctranslate2 sentencepiece
```

## Usage

Run the service:
```
python ct2_translate_service.py
```

The service communicates via JSON messages over stdin/stdout.

### Commands

- `{"cmd": "translate", "text": "Hello world", "source": "en", "target": "es"}` - Translate text
- `{"cmd": "list_available_models"}` - List all available models
- `{"cmd": "download_model", "lang_pair": "en-es"}` - Download and install a model
- `{"cmd": "delete_model", "lang_pair": "en-es"}` - Delete an installed model
- `{"cmd": "get_loaded_models"}` - Get list of loaded models
- `{"cmd": "health"}` - Health check
- `{"cmd": "shutdown"}` - Shutdown service

### Building Standalone Executable

Use PyInstaller:
```
python build.py
```

This creates a standalone executable in the `build/` directory.

## Models

Models are downloaded on-demand and stored locally. Pre-installed models are in `translation_models/`.

## License

See LICENSE file.