"""
Step 1 -- Data Acquisition & Merge

Loads every NHANES module for every cycle in config.CYCLES, left-joins each
module onto DEMO (the sampled-person universe) by SEQN, tags each row with
its cycle, and stacks all cycles into one raw merged file.

REBUILD NOTE: unchanged from the original pipeline except that config.MODULES
now includes "BPQ" (see config.py FIX note) -- this script itself is fully
generic over the module list, so no logic change was needed here to pull in
the new module; it "just worked" once BPQ was added to config.

Output: data_processed/nhanes_merged_raw.csv

Run from anywhere:
    python 06_Code/01_merge_data.py
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as cfg


def read_xpt(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        print(f"  [MISSING] {path.name} not found -- skipping this module for this cycle")
        return None
    try:
        df = pd.read_sas(path, format="xport")
    except Exception as e:
        print(f"  [ERROR] failed to read {path.name}: {e}")
        return None
    # FIX (2026-10 audit, C26): the SAS transport reader returns a stored
    # value of 0 as 5.4e-79 (an IBM floating-point artefact). Left alone, any
    # "== 0" test fails silently (e.g. a fasting weight of zero, or "never
    # drank in the past 12 months"). Snap these to exactly 0.
    num = df.select_dtypes("number")
    tiny = num.abs().lt(1e-30) & num.ne(0)
    if tiny.values.any():
        df[num.columns] = num.mask(tiny, 0.0)
    return df


def merge_one_cycle(letter: str, years: str) -> pd.DataFrame | None:
    folder = cfg.DATA_RAW_DIR / f"NHANES_{years}"
    if not folder.exists():
        print(f"[SKIP] cycle {years}: folder not found at {folder}")
        return None

    print(f"\n=== Cycle {years} (letter {letter}) ===")

    anchor_path = folder / f"{cfg.ANCHOR_MODULE}_{letter}.xpt"
    anchor_df = read_xpt(anchor_path)
    if anchor_df is None:
        print(f"[SKIP] cycle {years}: anchor module {cfg.ANCHOR_MODULE} missing, cannot merge")
        return None
    print(f"  {cfg.ANCHOR_MODULE}_{letter}.xpt: {anchor_df.shape[0]} rows, {anchor_df.shape[1]} cols")

    merged = anchor_df.copy()
    n_before_join = merged.shape[0]

    for module in cfg.MODULES:
        if module == cfg.ANCHOR_MODULE:
            continue
        path = folder / f"{module}_{letter}.xpt"
        df = read_xpt(path)
        if df is None:
            continue
        # avoid duplicate non-key columns across modules (rare, but be safe)
        overlap = (set(df.columns) & set(merged.columns)) - {cfg.ID_VAR}
        if overlap:
            print(f"  [NOTE] {module}_{letter}.xpt shares columns with merged data, dropping dupes: {sorted(overlap)}")
            df = df.drop(columns=list(overlap))
        merged = merged.merge(df, on=cfg.ID_VAR, how="left")
        print(f"  {module}_{letter}.xpt: {df.shape[0]} rows, {df.shape[1]} cols -> merged now {merged.shape[0]} rows, {merged.shape[1]} cols")

    if merged.shape[0] != n_before_join:
        print(f"  [WARNING] row count changed during left-join merge ({n_before_join} -> {merged.shape[0]}); "
              f"a non-anchor module likely has duplicate SEQNs")

    merged[cfg.CYCLE_VAR] = years

    # within-cycle SEQN should be unique (it's the sampled-person ID for that cycle)
    dupes = merged[cfg.ID_VAR].duplicated().sum()
    if dupes:
        print(f"  [WARNING] {dupes} duplicate SEQN within cycle {years} -- dropping duplicates, keeping first")
        merged = merged.drop_duplicates(subset=cfg.ID_VAR, keep="first")

    return merged


def main():
    all_cycles = []
    for cyc in cfg.CYCLES:
        df = merge_one_cycle(cyc["letter"], cyc["years"])
        if df is not None:
            all_cycles.append(df)

    if not all_cycles:
        raise SystemExit("No cycles were successfully loaded. Check NHANES_data/ folder structure.")

    print("\n=== Stacking all cycles ===")
    full = pd.concat(all_cycles, ignore_index=True, sort=False)
    print(f"Combined: {full.shape[0]} rows, {full.shape[1]} columns across {len(all_cycles)} cycles")

    # SEQN is NOT globally unique across cycles (NHANES reuses the ID space each
    # cycle) -- that's expected. Verify uniqueness of (SEQN, cycle) instead.
    key_dupes = full.duplicated(subset=[cfg.ID_VAR, cfg.CYCLE_VAR]).sum()
    print(f"Duplicate (SEQN, cycle) pairs after stacking: {key_dupes} (should be 0)")

    print("\nRows per cycle:")
    print(full.groupby(cfg.CYCLE_VAR).size().to_string())

    out_path = cfg.DATA_PROC_DIR / "nhanes_merged_raw.csv"
    full.to_csv(out_path, index=False)
    print(f"\nSaved: {out_path}  ({full.shape[0]} rows, {full.shape[1]} cols)")


if __name__ == "__main__":
    main()
