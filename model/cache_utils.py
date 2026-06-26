from pathlib import Path
import pandas as pd

# ── Global state ──────────────────────────────────────────────────────────────

cache_on: bool = True
_memory_cache: dict = {}

# Default cache directory relative to your project root
DEFAULT_CACHE_DIR = Path(__file__).parent / "data" / ".cache"


# ── Toggle controls ───────────────────────────────────────────────────────────

def activate_cache() -> None:
    """Enable caching globally."""
    global cache_on
    cache_on = True
    print("Cache enabled.")

def deactivate_cache() -> None:
    """Disable caching globally."""
    global cache_on
    cache_on = False
    print("Cache disabled.")

def is_cache_active() -> bool:
    """Return current cache state."""
    return cache_on 


# ── Memory cache (in-session) ─────────────────────────────────────────────────

def get_from_memory(key: str) -> pd.DataFrame | None:
    """Retrieve a DataFrame from the in-memory cache."""
    if cache_on and key in _memory_cache:
        print(f"[cache] Memory hit: {key}")
        return _memory_cache[key].copy()
    return None

def store_in_memory(key: str, df: pd.DataFrame) -> None:
    """Store a DataFrame in the in-memory cache."""
    if cache_on:
        _memory_cache[key] = df.copy()

def clear_memory_cache() -> None:
    """Clear the in-memory cache."""
    _memory_cache.clear()
    print("[cache] Memory cache cleared.")


# ── Disk cache (persists between sessions) ────────────────────────────────────

def get_from_disk(filepath: str | Path, cache_dir: str | Path = DEFAULT_CACHE_DIR) -> pd.DataFrame | None:
    """
    Retrieve a cached DataFrame from disk if it exists and is newer than the source file.
    Returns None on cache miss or stale cache.
    """
    if not cache_on:
        return None

    filepath = Path(filepath)
    cache_path = Path(cache_dir) / (filepath.stem + ".parquet")

    if not cache_path.exists():
        return None

    # Invalidate if source file has been modified since caching
    if filepath.stat().st_mtime > cache_path.stat().st_mtime:
        print(f"[cache] Stale cache for {filepath.name}, reloading.")
        return None

    print(f"[cache] Disk hit: {cache_path}")
    return pd.read_parquet(cache_path)

def store_on_disk(df: pd.DataFrame, filepath: str | Path, cache_dir: str | Path = DEFAULT_CACHE_DIR) -> None:
    """Save a DataFrame to disk cache as parquet."""
    if not cache_on:
        return

    cache_path = Path(cache_dir) / (Path(filepath).stem + ".parquet")
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(cache_path)
    print(f"[cache] Saved to disk: {cache_path}")

def clear_disk_cache(cache_dir: str | Path = DEFAULT_CACHE_DIR) -> None:
    """Delete all parquet files in the cache directory."""
    cache_dir = Path(cache_dir)
    if not cache_dir.exists():
        print("[cache] No disk cache found.")
        return
    for f in cache_dir.glob("*.parquet"):
        f.unlink()
    print(f"[cache] Disk cache cleared: {cache_dir}")


# ── Full clear ────────────────────────────────────────────────────────────────

def clear_cache() -> None:
    """Clear both memory and disk caches."""
    clear_memory_cache()
    clear_disk_cache()