from __future__ import annotations

import json
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PROJECT_ROOT / ".cache" / "oric_repos"
EVE_ASSETS_ROOT = PROJECT_ROOT / ".cache" / "oric_eve_assets"

# EVE README: https://github.com/baaivision/EVE/blob/main/EVEv1/README.md
_EVE_GDRIVE_PREPROCESSORS: dict[str, str] = {
    "eve-patch14-anypixel-672": "1f_mA4owjm0v3awrzPv4LOURz6IzVFVZ6",
    "eve-patch14-anypixel-1344": "1V7hz37X7n9s2KmghoQ9bDVHE6J4HuQ7z",
}
_EVE_HF_PREPROCESSORS: dict[str, str] = {
    "clip-vit-large-patch14-336": "openai/clip-vit-large-patch14-336",
}
_EVE_CONFIG_PREPROCESSOR_KEYS = ("mm_vision_tower", "mm_vision_tower_clip")

_REPO_URLS = {
    "VILA": "https://github.com/NVLabs/VILA.git",
    "Janus": "https://github.com/deepseek-ai/Janus.git",
    "EVEv1": "https://github.com/baaivision/EVE.git",
}


def ensure_git_repo(name: str, url: str | None = None) -> Path:
    repo_path = REPO_ROOT / name
    if repo_path.exists():
        return repo_path
    repo_path.parent.mkdir(parents=True, exist_ok=True)
    clone_url = url or _REPO_URLS[name]
    subprocess.run(
        ["git", "clone", "--depth", "1", clone_url, str(repo_path)],
        check=True,
    )
    return repo_path


def _patch_collections_for_legacy_libs() -> None:
    """Janus/attrdict expect symbols on `collections`, not only `collections.abc`."""
    import collections
    import collections.abc

    for type_name in collections.abc.__all__:
        if not hasattr(collections, type_name):
            setattr(collections, type_name, getattr(collections.abc, type_name))


def _patch_eve_builder_4bit(repo_path: Path) -> None:
    """EVE builder passes load_in_4bit and quantization_config; transformers rejects both."""
    builder_py = repo_path / "EVEv1" / "eve" / "model" / "builder.py"
    if not builder_py.exists():
        return
    text = builder_py.read_text(encoding="utf-8")
    old = "    elif load_4bit:\n        kwargs['load_in_4bit'] = True\n        kwargs['quantization_config']"
    new = "    elif load_4bit:\n        kwargs['quantization_config']"
    if old in text:
        text = text.replace(old, new, 1)
        builder_py.write_text(text, encoding="utf-8")


def _patch_eve_builder_preprocessors(repo_path: Path) -> None:
    """Pass local preprocessor paths via config= into EVELlamaForCausalLM.from_pretrained."""
    builder_py = repo_path / "EVEv1" / "eve" / "model" / "builder.py"
    if not builder_py.exists():
        return
    text = builder_py.read_text(encoding="utf-8")
    if "# ORIC: eve preprocessor paths" in text:
        return
    old = """            tokenizer = AutoTokenizer.from_pretrained(
                model_path, use_fast=False)
            model = EVELlamaForCausalLM.from_pretrained(
                model_path, low_cpu_mem_usage=True, **kwargs)"""
    new = """            tokenizer = AutoTokenizer.from_pretrained(
                model_path, use_fast=False)
            # ORIC: eve preprocessor paths
            _oric_eve_cfg = None
            try:
                import importlib
                _oric_eve_cfg = importlib.import_module("utils.repo_setup").prepare_eve_config(model_path)
            except Exception:
                pass
            _oric_eve_load_kw = dict(kwargs)
            if _oric_eve_cfg is not None:
                _oric_eve_load_kw["config"] = _oric_eve_cfg
            model = EVELlamaForCausalLM.from_pretrained(
                model_path, low_cpu_mem_usage=True, **_oric_eve_load_kw)"""
    if old in text:
        builder_py.write_text(text.replace(old, new, 1), encoding="utf-8")


def _ensure_vila_import_deps() -> None:
    """VILA llava_arch imports hydra and loguru at import time."""
    missing: list[str] = []
    for module, pip_pkg in (("hydra", "hydra-core"), ("loguru", "loguru")):
        try:
            __import__(module)
        except ImportError:
            missing.append(pip_pkg)
    if missing:
        pkgs = " ".join(missing)
        raise ImportError(
            f"VILA requires: {pkgs}. Install with: pip install {pkgs}"
        )


def _preprocessor_ready(path: Path) -> bool:
    return (path / "preprocessor_config.json").exists() or (path / "config.json").exists()


def _openai_asset_dir(relative_path: str) -> Path:
    """Map `openai/eve-patch14-anypixel-1344` → `.cache/oric_eve_assets/openai/...`."""
    normalized = relative_path.replace("\\", "/").strip("/")
    if normalized.startswith("openai/"):
        normalized = normalized[len("openai/") :]
    return EVE_ASSETS_ROOT / "openai" / normalized


def _flatten_preprocessor_dir(dest: Path) -> None:
    if _preprocessor_ready(dest):
        return
    children = [p for p in dest.iterdir() if p.is_dir()]
    if len(children) != 1:
        return
    nested = children[0]
    for item in nested.iterdir():
        target = dest / item.name
        if target.exists():
            continue
        shutil.move(str(item), str(target))
    shutil.rmtree(nested, ignore_errors=True)


def _download_hf_preprocessor(repo_id: str, dest: Path) -> None:
    from huggingface_hub import snapshot_download

    dest.mkdir(parents=True, exist_ok=True)
    snapshot_download(repo_id=repo_id, local_dir=str(dest))


def _download_gdrive_preprocessor(name: str, dest: Path) -> None:
    file_id = _EVE_GDRIVE_PREPROCESSORS.get(name)
    if file_id is None:
        raise ValueError(f"No Google Drive mapping for EVE preprocessor: {name}")

    try:
        import gdown
    except ImportError as exc:
        raise ImportError(
            "EVE Google Drive preprocessors require gdown. Install with: pip install gdown"
        ) from exc

    dest.mkdir(parents=True, exist_ok=True)
    archive = dest.parent / f"{name}.download"
    gdown.download(id=file_id, output=str(archive), quiet=False)

    if zipfile.is_zipfile(archive):
        with zipfile.ZipFile(archive, "r") as zf:
            zf.extractall(dest)
        archive.unlink(missing_ok=True)
        _flatten_preprocessor_dir(dest)
        return

    # Some Drive files are single JSON checkpoints without zip.
    if archive.suffix != ".json":
        target = dest / "preprocessor_config.json"
        if not target.exists():
            shutil.move(str(archive), str(target))
        return
    shutil.move(str(archive), str(dest / archive.name))


def ensure_eve_preprocessor_path(relative_path: str) -> Path:
    """Ensure one EVE preprocessor exists locally; return absolute path."""
    if not relative_path:
        raise ValueError("Empty EVE preprocessor path")

    local_path = Path(relative_path)
    if local_path.is_dir() and _preprocessor_ready(local_path):
        return local_path.resolve()

    if local_path.is_file():
        return local_path.resolve()

    asset_name = relative_path.replace("\\", "/").strip("/")
    if asset_name.startswith("openai/"):
        asset_name = asset_name[len("openai/") :]

    dest = _openai_asset_dir(relative_path)
    if _preprocessor_ready(dest):
        return dest.resolve()

    print(f"[eve] Downloading preprocessor: {relative_path} -> {dest}")
    dest.mkdir(parents=True, exist_ok=True)

    if asset_name in _EVE_GDRIVE_PREPROCESSORS:
        _download_gdrive_preprocessor(asset_name, dest)
    elif asset_name in _EVE_HF_PREPROCESSORS:
        _download_hf_preprocessor(_EVE_HF_PREPROCESSORS[asset_name], dest)
    elif asset_name.startswith("clip-") or asset_name.startswith("openai/"):
        repo_id = relative_path if "/" in relative_path else f"openai/{asset_name}"
        _download_hf_preprocessor(repo_id, dest)
    else:
        raise FileNotFoundError(
            f"Unknown EVE preprocessor '{relative_path}'. "
            f"Supported Google Drive names: {sorted(_EVE_GDRIVE_PREPROCESSORS)}; "
            f"HF names: {sorted(_EVE_HF_PREPROCESSORS)}."
        )

    if not _preprocessor_ready(dest):
        raise FileNotFoundError(
            f"Downloaded EVE preprocessor at {dest} is missing preprocessor_config.json."
        )
    return dest.resolve()


def prepare_eve_config(model_name_or_path: str) -> Any:
    """Load EVE config from HF and rewrite preprocessor paths to local absolute paths."""
    from huggingface_hub import hf_hub_download
    from transformers import AutoConfig

    config_path = hf_hub_download(model_name_or_path, "config.json")
    with open(config_path, encoding="utf-8") as f:
        config_dict = json.load(f)

    for key in _EVE_CONFIG_PREPROCESSOR_KEYS:
        value = config_dict.get(key)
        if not value:
            continue
        local = ensure_eve_preprocessor_path(str(value))
        config_dict[key] = str(local)
        print(f"[eve] {key}: {value} -> {local}")

    config = AutoConfig.from_pretrained(model_name_or_path, trust_remote_code=True)
    for key in _EVE_CONFIG_PREPROCESSOR_KEYS:
        if key in config_dict:
            setattr(config, key, config_dict[key])
    return config


def register_eve_builder_hooks() -> None:
    """Ensure ORIC root is importable when patched EVE builder loads preprocessors."""
    root = str(PROJECT_ROOT.resolve())
    if root not in sys.path:
        sys.path.insert(1, root)


def add_repo_to_path(name: str, *, subdir: str = "") -> Path:
    repo_path = ensure_git_repo(name)
    if name == "EVEv1":
        _patch_eve_builder_4bit(repo_path)
        _patch_eve_builder_preprocessors(repo_path)
    path = repo_path / subdir if subdir else repo_path
    path_str = str(path.resolve())
    if path_str not in sys.path:
        sys.path.insert(0, path_str)
    if name == "Janus":
        _patch_collections_for_legacy_libs()
    if name == "VILA":
        _ensure_vila_import_deps()
    return path
