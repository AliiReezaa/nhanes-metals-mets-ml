"""
Step 0 -- Download the raw NHANES files this study uses.

Downloads every module listed in config.MODULES for every cycle listed in
config.CYCLES from the CDC website and saves them where 01_merge_data.py
expects them:

    NHANES_data/NHANES_<years>/<MODULE>_<letter>.xpt

Examples: NHANES_data/NHANES_2007_2008/DEMO_E.xpt,
          NHANES_data/NHANES_2017_2018/PBCD_J.xpt

The script is safe to run many times: files that already exist and look like
valid SAS transport files are skipped, so an interrupted download simply
resumes. It uses only the Python standard library, so it works before any
other package is installed.

CDC moved its file server in 2024. The current address pattern is tried
first and the older pattern second, so the script keeps working if either
one is retired.

Run:
    python 06_Code/00_download_nhanes.py
"""
import sys
import time
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as cfg

URL_PATTERNS = [
    # current pattern, e.g. .../Public/2011/DataFiles/PBCD_G.xpt
    "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/{start}/DataFiles/{name}.xpt",
    # older pattern, e.g. .../Nhanes/2011-2012/PBCD_G.XPT
    "https://wwwn.cdc.gov/Nchs/Nhanes/{start}-{end}/{name}.XPT",
]
HEADERS = {"User-Agent": "Mozilla/5.0 (NHANES research download; python urllib)"}
MAGIC = b"HEADER RECORD"  # every SAS transport (.xpt) file starts with this


def looks_valid(path: Path) -> bool:
    if not path.exists() or path.stat().st_size < 1000:
        return False
    with open(path, "rb") as f:
        return MAGIC in f.read(200)


def fetch(url: str, dest: Path, tries: int = 3) -> bool:
    for attempt in range(1, tries + 1):
        try:
            req = urllib.request.Request(url, headers=HEADERS)
            with urllib.request.urlopen(req, timeout=120) as r:
                data = r.read()
            if MAGIC not in data[:200]:
                return False  # the server answered with a web page, not a data file
            tmp = dest.with_suffix(".part")
            tmp.write_bytes(data)
            tmp.replace(dest)
            return True
        except Exception as e:  # network hiccup: wait and retry
            if attempt == tries:
                print(f"      failed: {type(e).__name__}: {e}")
            time.sleep(3 * attempt)
    return False


def main():
    total = len(cfg.CYCLES) * len(cfg.MODULES)
    done = skipped = failed = 0
    missing = []
    for cyc in cfg.CYCLES:
        start, end = cyc["years"].split("_")
        folder = cfg.DATA_RAW_DIR / f"NHANES_{cyc['years']}"
        folder.mkdir(parents=True, exist_ok=True)
        for module in cfg.MODULES:
            name = f"{module}_{cyc['letter']}"
            dest = folder / f"{name}.xpt"
            if looks_valid(dest):
                skipped += 1
                continue
            print(f"  downloading {cyc['years']}  {name} ...")
            ok = any(fetch(p.format(start=start, end=end, name=name), dest) for p in URL_PATTERNS)
            if ok:
                done += 1
                print(f"      saved {dest.stat().st_size / 1e6:.1f} MB")
            else:
                failed += 1
                missing.append(f"{cyc['years']}/{name}")

    print(f"\nNHANES download: {done} downloaded, {skipped} already present, {failed} failed "
          f"(of {total} files).")
    if missing:
        print("Could not download:", ", ".join(missing))
        print("Download these by hand from https://wwwn.cdc.gov/nchs/nhanes/ and place each file in "
              "NHANES_data/NHANES_<years>/ with the exact name shown, then run again.")
        sys.exit(1)


if __name__ == "__main__":
    main()
