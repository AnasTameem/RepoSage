# ingestion/cloner.py
import subprocess
import shutil
from pathlib import Path


class CloneError(Exception):
    """Raised when cloning fails or repo violates size/time limits."""
    pass


def clone_repo(repo_url: str, dest_dir: str = "./cloned_repo",
               timeout: int = 600, max_size_mb: int = 500) -> Path:
    dest_path = Path(dest_dir)

    if dest_path.exists():
        shutil.rmtree(dest_path)  # purana clone hai toh saaf karo pehle

    try:
        subprocess.run(
            ["git", "clone", "--depth", "1", repo_url, str(dest_path)],
            check=True,
            timeout=timeout,
            capture_output=True,
            text=True,
        )
    except subprocess.TimeoutExpired:
        raise CloneError(f"Clone timed out after {timeout}s")
    except subprocess.CalledProcessError as e:
        raise CloneError(f"git clone failed: {e.stderr}")

    size_mb = _get_dir_size(dest_path) / (1024 * 1024)
    if size_mb > max_size_mb:
        shutil.rmtree(dest_path)
        raise CloneError(f"Repo too large: {size_mb:.1f}MB (limit {max_size_mb}MB)")

    return dest_path


def _get_dir_size(path: Path) -> int:
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file())


def cleanup_repo(dest_dir: str = "./cloned_repo") -> None:
    dest_path = Path(dest_dir)
    if dest_path.exists():
        shutil.rmtree(dest_path)