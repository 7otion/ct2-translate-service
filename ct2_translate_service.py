#!/usr/bin/env python3
"""
Production Translation Service with Model Download Support
Allows users to download and install language models interactively
"""
from __future__ import annotations
import sys
import os
import json
import asyncio
import signal
import logging
import time
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from typing import Optional, Dict, List
from urllib.request import urlopen, Request
from urllib.error import URLError, HTTPError

# Resource path for PyInstaller
def resource_path(rel: str) -> str:
    if hasattr(sys, "_MEIPASS"):
        return os.path.join(sys._MEIPASS, rel)
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), rel)

# Models directory - for bundled exe, use user's local data
if hasattr(sys, "_MEIPASS"):
    # Running as bundled exe - store models in user data directory
    if os.name == "nt":
        user_data = Path(os.environ.get("LOCALAPPDATA", os.path.expanduser("~"))) / "TranslationService" / "models"
    else:
        user_data = Path.home() / ".local" / "share" / "translation-service" / "models"
    MODELS_DIR = user_data
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
else:
    # Development mode - use local directory
    MODELS_DIR = Path(resource_path("translation_models"))

# Logging
logger = logging.getLogger("translate_service")
handler = logging.StreamHandler(sys.stderr)
handler.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
logger.addHandler(handler)
logger.setLevel(logging.WARNING)

if os.name == "nt":
    try:
        import msvcrt
        msvcrt.setmode(sys.stdout.fileno(), os.O_BINARY)
    except Exception:
        pass

def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()

def send_status(obj: dict) -> None:
    try:
        sys.stderr.write(json.dumps(obj, default=str) + "\n")
        sys.stderr.flush()
    except Exception:
        pass

def send_result(obj: dict) -> None:
    try:
        sys.stdout.write(json.dumps(obj, default=str) + "\n")
        sys.stdout.flush()
    except Exception:
        pass

# Check dependencies - lazy loaded in TranslationModel.load()
CT2_AVAILABLE = True
SP_AVAILABLE = True
ARGOS_AVAILABLE = False


# Available language pairs from Argos repository
AVAILABLE_MODELS = {
    "en-es": {"name": "English → Spanish", "from": "en", "to": "es"},
    "es-en": {"name": "Spanish → English", "from": "es", "to": "en"},
    "en-fr": {"name": "English → French", "from": "en", "to": "fr"},
    "fr-en": {"name": "French → English", "from": "fr", "to": "en"},
    "en-de": {"name": "English → German", "from": "en", "to": "de"},
    "de-en": {"name": "German → English", "from": "de", "to": "en"},
    "en-tr": {"name": "English → Turkish", "from": "en", "to": "tr"},
    "tr-en": {"name": "Turkish → English", "from": "tr", "to": "en"},
    "en-ar": {"name": "English → Arabic", "from": "en", "to": "ar"},
    "ar-en": {"name": "Arabic → English", "from": "ar", "to": "en"},
    "en-zh": {"name": "English → Chinese", "from": "en", "to": "zh"},
    "zh-en": {"name": "Chinese → English", "from": "zh", "to": "en"},
    "en-ru": {"name": "English → Russian", "from": "en", "to": "ru"},
    "ru-en": {"name": "Russian → English", "from": "ru", "to": "en"},
    "en-pt": {"name": "English → Portuguese", "from": "en", "to": "pt"},
    "pt-en": {"name": "Portuguese → English", "from": "pt", "to": "en"},
    "en-it": {"name": "English → Italian", "from": "en", "to": "it"},
    "it-en": {"name": "Italian → English", "from": "it", "to": "en"},
    "en-ja": {"name": "English → Japanese", "from": "en", "to": "ja"},
    "ja-en": {"name": "Japanese → English", "from": "ja", "to": "en"},
    "en-ko": {"name": "English → Korean", "from": "en", "to": "ko"},
    "ko-en": {"name": "Korean → English", "from": "ko", "to": "en"},
}

class TranslationModel:
    """Translation model wrapper"""
    
    def __init__(self, model_dir: Path, lang_pair: str):
        self.model_dir = model_dir
        self.lang_pair = lang_pair
        self.translator = None
        self.sp_processor = None
        
    def load(self):
        """Load model"""
        try:
            import ctranslate2
            import sentencepiece as spm
        except ImportError as e:
            raise RuntimeError(f"Missing dependencies: {e}")
        
        model_path = self.model_dir / "model"
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found at {model_path}")
        
        self.translator = ctranslate2.Translator(
            str(model_path),
            device="cpu",
            compute_type="int8",
            inter_threads=2,
            intra_threads=4
        )
        
        sp_path = self.model_dir / "sentencepiece.model"
        if not sp_path.exists():
            raise FileNotFoundError(f"SentencePiece model not found")
        
        self.sp_processor = spm.SentencePieceProcessor(str(sp_path))
        logger.warning("Loaded %s", self.lang_pair)
    
    def translate(self, text: str) -> str:
        """Translate text"""
        if not self.translator or not self.sp_processor:
            raise RuntimeError("Model not loaded")
        
        source_tokens = self.sp_processor.encode(text, out_type=str)
        
        results = self.translator.translate_batch(
            [source_tokens],
            beam_size=1,
            repetition_penalty=1.2,
            no_repeat_ngram_size=2,
            max_decoding_length=256,
            min_decoding_length=1,
            return_scores=False
        )
        
        translated_tokens = results[0].hypotheses[0]
        token_ids = [self.sp_processor.piece_to_id(piece) for piece in translated_tokens]
        return self.sp_processor.decode(token_ids)

import tarfile
import zipfile
from urllib.request import urlopen, Request
from urllib.error import URLError, HTTPError

class ModelDownloader:
    """Download Argos .argosmodel packages directly from the public index and extract into MODELS_DIR."""
    def __init__(self, models_dir: Path):
        self.models_dir = models_dir
        self.models_dir.mkdir(parents=True, exist_ok=True)
        # store temp downloads inside models dir (safer in sandboxed packaged apps)
        self._temp_dir = self.models_dir / ".tmp_downloads"
        self._temp_dir.mkdir(exist_ok=True)
        # default index — raw GitHub index JSON (can be overridden by env var)
        self.index_url = os.environ.get(
            "MODEL_INDEX_URL",
            "https://raw.githubusercontent.com/argosopentech/argospm-index/main/index.json"
        )

    def get_available_models(self) -> List[Dict]:
        """Return the configured AVAILABLE_MODELS with installed flag (same as before)."""
        available = []
        for lang_pair, info in AVAILABLE_MODELS.items():
            model_dir = self.models_dir / lang_pair
            installed = model_dir.exists() and (model_dir / "model").exists()
            available.append({
                "lang_pair": lang_pair,
                "name": info["name"],
                "from_lang": info["from"],
                "to_lang": info["to"],
                "installed": installed
            })
        return available

    def _fetch_index(self) -> List[Dict]:
        """Fetch Argos package index JSON (list of packages)."""
        try:
            req = Request(self.index_url, headers={"User-Agent": "TranslationService/1.0"})
            with urlopen(req, timeout=30) as r:
                data = r.read().decode("utf-8")
            index = json.loads(data)
            # index.json in argospm-index is typically a list of package dicts
            return index
        except Exception as e:
            raise RuntimeError(f"Failed to fetch model index from {self.index_url}: {e}")

    def _choose_package_entry(self, index: List[Dict], from_code: str, to_code: str) -> Dict:
        """Find matching package entry in the index. Returns package dict or raises."""
        # Preference: exact from_code,to_code match, latest package_version if multiple
        candidates = [p for p in index if p.get("from_code") == from_code and p.get("to_code") == to_code]
        if not candidates:
            raise RuntimeError(f"No package found in index for {from_code}→{to_code}")
        # sort by package_version (if present) or return first
        try:
            candidates.sort(key=lambda p: p.get("package_version", ""), reverse=True)
        except Exception:
            pass
        return candidates[0]

    def download_and_install(self, lang_pair: str, progress_callback=None) -> bool:
        """Download & extract model by looking up the index and downloading the .argosmodel archive."""
        if lang_pair not in AVAILABLE_MODELS:
            raise ValueError(f"Unknown language pair: {lang_pair}")
        info = AVAILABLE_MODELS[lang_pair]
        from_code = info["from"]
        to_code = info["to"]
        dest_dir = self.models_dir / lang_pair

        def _progress(s, p):
            if progress_callback:
                progress_callback({"status": s, "progress": p})

        _progress("start", 0)
        # 1) fetch index
        index = self._fetch_index()
        _progress("fetched_index", 10)

        # 2) find package entry
        pkg = self._choose_package_entry(index, from_code, to_code)
        _progress("selected_package", 20)

        # 3) get download link
        # Argos index typically has 'links' (list) or 'download' field — handle both
        download_url = None
        if isinstance(pkg.get("links"), list) and pkg["links"]:
            download_url = pkg["links"][0]
        elif pkg.get("download"):
            download_url = pkg["download"]
        else:
            # some indexes embed different fields; inspect 'url' or 'link'
            for k in ("url", "link"):
                if pkg.get(k):
                    download_url = pkg[k]
                    break
        if not download_url:
            raise RuntimeError("No download link found in package metadata")

        _progress("downloading", 40)

        # 4) stream-download to temp file
        tmp_fname = self._temp_dir / f"{lang_pair}.archive"
        try:
            req = Request(download_url, headers={"User-Agent": "TranslationService/1.0"})
            with urlopen(req, timeout=120) as resp, open(tmp_fname, "wb") as out_f:
                # stream in chunks for large files
                chunk_size = 64 * 1024
                while True:
                    chunk = resp.read(chunk_size)
                    if not chunk:
                        break
                    out_f.write(chunk)
        except Exception as e:
            raise RuntimeError(f"Failed to download {download_url}: {e}")

        _progress("downloaded", 70)

        # 5) extract archive to dest_dir
        self._extract_archive_to_model(tmp_fname, dest_dir)
        _progress("complete", 100)
        # cleanup tmp file
        try:
            tmp_fname.unlink()
        except Exception:
            pass
        return True

    def _extract_archive_to_model(self, archive_path: Path, dest_dir: Path):
        """Extract tar/zip/.argosmodel and copy model/ and sentencepiece.model into dest_dir."""
        if dest_dir.exists():
            shutil.rmtree(dest_dir)
        dest_dir.mkdir(parents=True, exist_ok=True)

        tmp_extract = self._temp_dir / f"{archive_path.stem}_extracted"
        if tmp_extract.exists():
            shutil.rmtree(tmp_extract)
        tmp_extract.mkdir(parents=True, exist_ok=True)

        try:
            if tarfile.is_tarfile(archive_path):
                with tarfile.open(archive_path, "r:*") as tf:
                    tf.extractall(path=tmp_extract)
            elif zipfile.is_zipfile(archive_path):
                with zipfile.ZipFile(archive_path, "r") as zf:
                    zf.extractall(path=tmp_extract)
            else:
                raise RuntimeError("Unknown or unsupported archive format for package")
        except Exception as ex:
            raise RuntimeError(f"Failed to extract archive {archive_path}: {ex}")

        # Find model folder (recursively)
        model_src = None
        sp_src = None
        for p in tmp_extract.rglob("model"):
            if p.is_dir():
                model_src = p
                break
        for p in tmp_extract.rglob("sentencepiece.model"):
            sp_src = p
            break

        # fallback: some packages wrap things under a top-level folder
        if not model_src:
            for root in tmp_extract.iterdir():
                candidate = root / "model"
                if candidate.exists():
                    model_src = candidate
                    break

        if not model_src:
            shutil.rmtree(tmp_extract)
            raise FileNotFoundError("model/ directory not found inside the downloaded package")

        shutil.copytree(model_src, dest_dir / "model")

        if sp_src and sp_src.exists():
            shutil.copy(sp_src, dest_dir / "sentencepiece.model")
        else:
            # maybe SentencePiece was named differently — fail explicitly
            raise FileNotFoundError("sentencepiece.model not found inside the downloaded package")

        # cleanup
        try:
            shutil.rmtree(tmp_extract)
        except Exception:
            pass

        logger.warning("Extracted model to %s", dest_dir)

    def delete_model(self, lang_pair: str) -> bool:
        model_dir = self.models_dir / lang_pair
        if model_dir.exists():
            shutil.rmtree(model_dir)
            return True
        return False
    
    def _extract_ct2_model(self, argos_dir: Path, dest_dir: Path):
        """Extract CT2 model files from Argos package"""
        # Clean destination
        if dest_dir.exists():
            shutil.rmtree(dest_dir)
        dest_dir.mkdir(parents=True)
        
        # Copy model directory
        model_src = argos_dir / "model"
        if not model_src.exists():
            raise FileNotFoundError(f"Model directory not found in {argos_dir}")
        
        shutil.copytree(model_src, dest_dir / "model")
        
        # Copy SentencePiece model
        sp_src = argos_dir / "sentencepiece.model"
        if sp_src.exists():
            shutil.copy(sp_src, dest_dir / "sentencepiece.model")
        else:
            raise FileNotFoundError("SentencePiece model not found")
        
        logger.warning("Extracted CT2 model to %s", dest_dir)
    
    def delete_model(self, lang_pair: str) -> bool:
        """Delete an installed model"""
        model_dir = self.models_dir / lang_pair
        if model_dir.exists():
            shutil.rmtree(model_dir)
            return True
        return False

class TranslateService:
    """Translation service with model management"""
    
    def __init__(self, models_dir: Path):
        self._running = True
        self._models_dir = models_dir
        self._models = {}
        self._lock = asyncio.Lock()
        self._executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="translate")
        self._downloader = ModelDownloader(models_dir)
        
        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                signal.signal(sig, self._on_signal)
            except Exception:
                pass
    
    async def _discover_and_load_models_async(self):
        """Run the blocking discovery in a threadpool and populate self._models."""
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(self._executor, self._discover_and_load_models)
    
    def _discover_and_load_models(self):
        """Discover and pre-load all models"""
        if not self._models_dir.exists():
            logger.warning("Models directory not found: %s", self._models_dir)
            return
        
        for model_dir in self._models_dir.iterdir():
            if not model_dir.is_dir():
                continue
            
            lang_pair = model_dir.name
            
            # Skip hidden or temporary directories
            if lang_pair.startswith('.'):
                continue
            
            try:
                send_status({"status": "loading_model", "lang_pair": lang_pair})
                model = TranslationModel(model_dir, lang_pair)
                model.load()
                self._models[lang_pair] = model
                logger.warning("Model ready: %s", lang_pair)
                send_status({"status": "model_loaded", "lang_pair": lang_pair})
            except Exception as e:
                logger.error("Failed to load %s: %s", lang_pair, e)
                send_status({"status": "model_load_failed", "lang_pair": lang_pair, "error": str(e)})
    
    async def run(self):
        """Main service loop"""
        # Announce ready immediately so the parent process doesn't block.
        send_status({
            "status": "ready",
            "ts": now_iso(),
            "loaded_models": list(self._models.keys()),
            "models_dir": str(self._models_dir),
            "version": "2.0.0"
        })

        # Load models in background so ready is immediate and long work doesn't block.
        # Use executor so blocking imports / model loads don't block the event loop.
        loop = asyncio.get_running_loop()
        loop.create_task(self._discover_and_load_models_async())

        # Now continue normal message loop
        while self._running:
            try:
                raw = await loop.run_in_executor(None, sys.stdin.buffer.readline)
                if not raw:
                    break
                
                line_s = raw.rstrip(b"\r\n").decode("utf-8", errors="replace")
                if not line_s:
                    continue
                
                try:
                    msg = json.loads(line_s)
                except json.JSONDecodeError as e:
                    send_status({"status": "error", "message": f"invalid json: {e}"})
                    continue
                
                cmd = msg.get("cmd")
                
                if cmd == "translate":
                    asyncio.create_task(self._handle_translate(msg))
                elif cmd == "list_available_models":
                    await self._handle_list_available(msg)
                elif cmd == "download_model":
                    asyncio.create_task(self._handle_download(msg))
                elif cmd == "delete_model":
                    await self._handle_delete(msg)
                elif cmd == "get_loaded_models":
                    send_result({"models": list(self._models.keys())})
                elif cmd == "health":
                    send_result({"status": "ok", "models_loaded": len(self._models)})
                elif cmd == "shutdown":
                    await self.shutdown()
                    break
                else:
                    send_status({"status": "error", "message": f"unknown cmd: {cmd}"})
            
            except Exception as e:
                logger.exception("Error in main loop: %s", e)
        
        await self.shutdown()
    
    async def _handle_translate(self, msg: dict):
        """Handle translation request"""
        req_id = msg.get("id")
        text = msg.get("text", "")
        source = msg.get("source", "en")
        target = msg.get("target")
        
        if not target:
            send_result({"id": req_id, "error": "target language required"})
            return
        
        if not text:
            send_result({"id": req_id, "translated": "", "latency_ms": 0})
            return
        
        request_ts = time.time()
        
        async with self._lock:
            try:
                lang_pair = f"{source}-{target}"
                model = self._models.get(lang_pair)
                
                if not model:
                    raise RuntimeError(f"Model not loaded: {lang_pair}. Use 'download_model' to install it.")
                
                loop = asyncio.get_running_loop()
                translated = await loop.run_in_executor(self._executor, model.translate, text)
                
                latency_ms = int((time.time() - request_ts) * 1000)
                send_result({"id": req_id, "translated": translated, "latency_ms": latency_ms})
            
            except Exception as e:
                logger.error("Translation failed: %s", e)
                send_result({"id": req_id, "error": str(e)})
    
    async def _handle_list_available(self, msg: dict):
        """List all available models (installed and not installed)"""
        req_id = msg.get("id")
        try:
            models = self._downloader.get_available_models()
            send_result({"id": req_id, "models": models})
        except Exception as e:
            send_result({"id": req_id, "error": str(e)})
    
    async def _handle_download(self, msg: dict):
        """Download and install a model"""
        req_id = msg.get("id")
        lang_pair = msg.get("lang_pair")
        
        if not lang_pair:
            send_result({"id": req_id, "error": "lang_pair required"})
            return
        
        def progress_callback(update: dict):
            send_status({"download_progress": update, "id": req_id})
        
        try:
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(
                self._executor,
                self._downloader.download_and_install,
                lang_pair,
                progress_callback
            )
            
            # Reload the model
            model_dir = self._models_dir / lang_pair
            model = TranslationModel(model_dir, lang_pair)
            model.load()
            
            async with self._lock:
                self._models[lang_pair] = model
            
            send_result({"id": req_id, "success": True, "lang_pair": lang_pair})
        
        except Exception as e:
            logger.error("Download failed: %s", e)
            send_result({"id": req_id, "error": str(e)})
    
    async def _handle_delete(self, msg: dict):
        """Delete an installed model"""
        req_id = msg.get("id")
        lang_pair = msg.get("lang_pair")
        
        if not lang_pair:
            send_result({"id": req_id, "error": "lang_pair required"})
            return
        
        try:
            # Remove from loaded models
            async with self._lock:
                if lang_pair in self._models:
                    del self._models[lang_pair]
            
            # Delete files
            loop = asyncio.get_running_loop()
            deleted = await loop.run_in_executor(
                self._executor,
                self._downloader.delete_model,
                lang_pair
            )
            
            send_result({"id": req_id, "success": deleted, "lang_pair": lang_pair})
        
        except Exception as e:
            send_result({"id": req_id, "error": str(e)})
    
    async def shutdown(self):
        """Clean shutdown"""
        if not self._running:
            return
        self._running = False
        self._executor.shutdown(wait=False)
        send_status({"status": "shutdown", "ts": now_iso()})

def main():
    svc = TranslateService(models_dir=MODELS_DIR)
    try:
        asyncio.run(svc.run())
    except KeyboardInterrupt:
        pass
    except Exception:
        logger.exception("Fatal error")
    finally:
        try:
            sys.stderr.flush()
            sys.stdout.flush()
        except Exception:
            pass

if __name__ == "__main__":
    main()