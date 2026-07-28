"""Pre-publication checks for the COBICount repository."""

from __future__ import annotations

from pathlib import Path
import compileall
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
SKIP_DIRS = {".git", ".venv", "venv", "__pycache__", ".pytest_cache", "build", "dist"}
FORBIDDEN_SUFFIXES = {".pth", ".pt", ".ckpt", ".onnx", ".mat", ".npy", ".npz"}
SENSITIVE_PATTERNS = {
    "original local project path": re.compile(r"[A-Za-z]:\\ProgramData\\AI\\Project", re.IGNORECASE),
    "possible GitHub token": re.compile(r"gh[pousr]_[A-Za-z0-9_]{20,}"),
    "possible AWS access key": re.compile(r"AKIA[0-9A-Z]{16}"),
}


def iter_files():
    for path in ROOT.rglob("*"):
        if not path.is_file():
            continue
        if any(part in SKIP_DIRS for part in path.parts):
            continue
        yield path


def main() -> int:
    errors: list[str] = []
    warnings: list[str] = []

    for path in iter_files():
        relative = path.relative_to(ROOT)
        if path.suffix.lower() in FORBIDDEN_SUFFIXES:
            errors.append(f"binary/data artifact should not be committed: {relative}")
        if path.stat().st_size > 50 * 1024 * 1024:
            errors.append(f"file exceeds 50 MiB release-check threshold: {relative}")
        if path.suffix.lower() in {".py", ".md", ".toml", ".yml", ".yaml", ".txt", ".cff", ".ps1", ".sh"}:
            text = path.read_text(encoding="utf-8", errors="ignore")
            for label, pattern in SENSITIVE_PATTERNS.items():
                if pattern.search(text):
                    errors.append(f"{label}: {relative}")
            if "OWNER/COBICount" in text:
                warnings.append(f"replace GitHub OWNER placeholder: {relative}")

    train_source = ROOT / "src" / "cobicount" / "cli" / "train.py"
    train_text = train_source.read_text(encoding="utf-8")
    for target_name in ["DOTAProxyPositiveEvalDataset", "DOTAFullImageEvalDataset"]:
        if target_name in train_text:
            errors.append(f"target-domain dataset leaked into training entry point: {target_name}")

    if not compileall.compile_dir(ROOT / "src", quiet=1):
        errors.append("Python compilation failed under src/")

    for warning in sorted(set(warnings)):
        print(f"[WARNING] {warning}")
    for error in sorted(set(errors)):
        print(f"[ERROR] {error}")

    if errors:
        print(f"Release check failed with {len(set(errors))} error(s).")
        return 1
    print("Release check passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
