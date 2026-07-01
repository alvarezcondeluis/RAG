"""
EDGAR Cache Solution - Working Version

The issue: use_local_storage() enables caching but edgartools may not persist
all data to disk by default. The library caches in memory for performance.

Solution: We need to explicitly download and save the filing HTML/text to disk.
"""
import time
from pathlib import Path
from edgar import Company, set_identity
from tqdm import tqdm
import pickle

# Configuration
set_identity("luis.alvarez.conde@alumnos.upm.es")
CACHE_DIR = Path("./edgar_cache")
CACHE_DIR.mkdir(exist_ok=True)

def get_cache_path(ticker: str, form: str, accession: str) -> Path:
    """Generate a cache file path for a filing."""
    # Create subdirectory for the ticker
    ticker_dir = CACHE_DIR / ticker
    ticker_dir.mkdir(exist_ok=True)
    
    # Sanitize accession number for filename
    safe_accession = accession.replace("/", "_")
    return ticker_dir / f"{form}_{safe_accession}.pkl"

def is_cached(ticker: str, form: str, accession: str) -> bool:
    """Check if a filing is already cached (and not empty/corrupt)."""
    p = get_cache_path(ticker, form, accession)
    return p.exists() and p.stat().st_size > 0

def save_to_cache(ticker: str, form: str, accession: str, filing_obj):
    """Save a filing object to disk cache using an atomic write (tmp → rename)."""
    cache_path = get_cache_path(ticker, form, accession)
    tmp = cache_path.with_suffix(".tmp")
    try:
        with open(tmp, 'wb') as f:
            pickle.dump(filing_obj, f)
        tmp.rename(cache_path)  # ponytail: atomic on Linux/macOS; avoids corrupt half-writes
    except Exception:
        tmp.unlink(missing_ok=True)
        raise

def load_from_cache(ticker: str, form: str, accession: str):
    """Load a filing object from disk cache. Deletes and returns None if corrupt."""
    cache_path = get_cache_path(ticker, form, accession)
    try:
        with open(cache_path, 'rb') as f:
            return pickle.load(f)
    except (pickle.UnpicklingError, EOFError, Exception):
        cache_path.unlink(missing_ok=True)
        return None

def warm_cache_with_persistence(tickers: list):
    """
    Download and cache EDGAR filings with explicit disk persistence.
    Clears the entire cache directory before downloading to ensure fresh data.
    """
    import shutil

    # Wipe existing cache so stale 10-K/A amendments are not reused
    if CACHE_DIR.exists():
        shutil.rmtree(CACHE_DIR)
        print(f"🗑️  Cleared existing cache at {CACHE_DIR.absolute()}")
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Starting download for {len(tickers)} companies...")
    print(f"Cache directory: {CACHE_DIR.absolute()}")

    stats = {
        "downloaded": 0,
        "cached": 0,
        "errors": 0
    }

    for ticker in tqdm(tickers, desc="Companies"):
        try:
            company = Company(ticker)

            # --- Download 10-K (original only — skip 10-K/A amendments) ---
            all_tenk = company.get_filings(form="10-K")
            original_tenks = [f for f in all_tenk if f.form.upper() == "10-K"]
            if not original_tenks:
                print(f"[{ticker}] ⚠️  No original 10-K found (only amendments?) — skipping 10-K")
                stats["errors"] += 1
            else:
                tenk = original_tenks[0]  # newest-first order
                if is_cached(ticker, "10-K", tenk.accession_number):
                    print(f"[{ticker}] 10-K already cached")
                    stats["cached"] += 1
                else:
                    obj = tenk.obj()
                    save_to_cache(ticker, "10-K", tenk.accession_number, obj)
                    print(f"[{ticker}] 10-K downloaded and cached")
                    stats["downloaded"] += 1
                    time.sleep(0.1)  # Be nice to SEC

            # --- Download DEF 14A (Proxy) ---
            def14a = company.get_filings(form="DEF 14A").latest()
            if def14a:
                if is_cached(ticker, "DEF14A", def14a.accession_number):
                    print(f"[{ticker}] DEF 14A already cached")
                    stats["cached"] += 1
                else:
                    obj = def14a.obj()
                    save_to_cache(ticker, "DEF14A", def14a.accession_number, obj)
                    print(f"[{ticker}] DEF 14A downloaded and cached")
                    stats["downloaded"] += 1
                    time.sleep(0.1)

            # --- Download Form 4s (Insider Trading) ---
            ins_filings = company.get_filings(form="4").filter(date='2025-09-01:')
            
            if len(ins_filings) > 0:
                cached_count = 0
                downloaded_count = 0
                
                for filing in ins_filings:
                    try:
                        if is_cached(ticker, "Form4", filing.accession_number):
                            cached_count += 1
                        else:
                            obj = filing.obj()
                            save_to_cache(ticker, "Form4", filing.accession_number, obj)
                            downloaded_count += 1
                            time.sleep(0.1)
                    except Exception as e:
                        print(f"  Skipped one Form 4: {e}")
                        stats["errors"] += 1
                
                if downloaded_count > 0:
                    print(f"[{ticker}] Downloaded {downloaded_count} Form 4s")
                    stats["downloaded"] += downloaded_count
                if cached_count > 0:
                    print(f"[{ticker}] {cached_count} Form 4s already cached")
                    stats["cached"] += cached_count
            
        except Exception as e:
            print(f"Error processing {ticker}: {e}")
            stats["errors"] += 1
    
    # Print summary
    print("\n" + "="*60)
    print("CACHE SUMMARY")
    print("="*60)
    print(f"Files downloaded: {stats['downloaded']}")
    print(f"Files already cached: {stats['cached']}")
    print(f"Errors: {stats['errors']}")
    
    # Verify cache
    total_files = sum(1 for _ in CACHE_DIR.rglob("*.pkl"))
    print(f"Total files in cache: {total_files}")
    print(f"Cache size: {sum(f.stat().st_size for f in CACHE_DIR.rglob('*.pkl')) / 1024 / 1024:.2f} MB")
    print("="*60)

# Example usage
if __name__ == "__main__":
    tickers = [
        'AAPL', 'MSFT', 'GOOG', 'AMZN', 'TSLA', 
        'NVDA', 'META', 'NFLX', 'DIS', 'KO', 
        'PEP', 'COST', 'WMT', 'NKE', 'SBUX', 
        'PFE', 'JNJ', 'XOM', 'V', 'PYPL'
    ]
    
    warm_cache_with_persistence(tickers)
