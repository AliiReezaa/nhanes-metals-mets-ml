#!/usr/bin/env python3
"""
RUN_ME.py -- run the whole machine-learning pipeline with one click.

What it does, in order
----------------------
1. Checks that a suitable Python (3.11 or newer) is running.
2. Creates a private Python environment in the folder ".venv" next to this
   file and installs every package the pipeline needs (first run only).
3. Asks which run you want (just press Enter for the default):
      1  FULL      download the raw NHANES files from the CDC and run every step
      2  PACKAGED  start from the cleaned data shipped in 05_Data_Processed/
                   (no internet needed after the packages are installed)
      3  QUICK     like PACKAGED, but with 3-fold instead of 10-fold searches;
                   a fast test drive (numbers will differ slightly)
4. Runs every numbered script in 06_Code/ in the correct order, stopping
   with a clear message if a required step fails.
5. Optionally reruns the exploratory analyses in 07_Novelty_Additions/ and
   08_Q1_Rigor_Additions/.
6. Compares the fresh results with the published ones and writes
   outputs/reproduction_report.txt.

Everything the run creates goes into new folders (NHANES_data/,
data_processed/, outputs/, manuscript/); the original package files are
never modified. A full log is saved to outputs/run_log.txt.

How to start it
---------------
  Windows : double-click RUN_ME_Windows.bat
  macOS   : double-click RUN_ME_macOS.command
  Linux   : run ./RUN_ME_Linux.sh   (or: python3 RUN_ME.py)
  Any OS  : open a terminal in this folder and type  python RUN_ME.py

Advanced: skip the question with  python RUN_ME.py --mode packaged
          (modes: full, packaged, quick; add --no-extras to skip step 5)
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
import time
import venv
from pathlib import Path

ROOT = Path(__file__).resolve().parent
CODE = ROOT / "06_Code"
VENV = ROOT / ".venv"
OUT = ROOT / "outputs"
LOG = OUT / "run_log.txt"
MIN_PY = (3, 11)

# Packages: the pinned list in 06_Code/requirements.txt plus the few extra
# libraries some scripts import but the original list left out.
EXTRA_PACKAGES = ["seaborn", "patsy", "plotly", "python-docx"]

# (label, script, extra arguments per mode, required?)
# Required steps stop the run when they fail; optional ones only warn.
CORE_STEPS = [
    ("Step 0  download raw NHANES files",          "00_download_nhanes.py",        {},                              True,  {"full"}),
    ("Step 1  merge modules and cycles",            "01_merge_data.py",             {},                              True,  {"full"}),
    ("Step 2-3 clean data, define MetS outcome",    "02_clean_and_outcome.py",      {},                              True,  {"full"}),
    ("Step 4  encode predictors",                   "03_feature_engineering.py",    {},                              True,  {"full", "packaged", "quick"}),
    ("Step 5  train/test split",                    "04_split.py",                  {},                              True,  {"full", "packaged", "quick"}),
    ("Step 8  cross-validated tuning (slowest)",    "06_cv_tuning.py",              {"quick": ["--quick"]},          True,  {"full", "packaged", "quick"}),
    ("Step 9-10 final models and thresholds",       "07_train_models.py",           {"quick": ["--cv", "3"]},        True,  {"full", "packaged", "quick"}),
    ("Step 11 evaluate on the test set",            "08_evaluate.py",               {},                              True,  {"full", "packaged", "quick"}),
    ("Step 12 SHAP and permutation importance",     "09_explain_shap.py",           {},                              True,  {"full", "packaged", "quick"}),
    ("Step 13 sensitivity analyses",                "10_sensitivity.py",            {"quick": ["--cv", "3"]},        True,  {"full", "packaged", "quick"}),
    ("Step 14 STROBE flow and Table 1",             "11_reporting.py",              {},                              True,  {"full", "packaged", "quick"}),
    ("Extra   comparison charts and dashboard",     "12_visualize.py",              {},                              False, {"full", "packaged", "quick"}),
    ("Step 15 publication figures and numbers",     "15_publication_figures.py",    {},                              True,  {"full", "packaged", "quick"}),
    ("Docs    write Methods .docx",                 "16_write_methods_docx.py",     {},                              False, {"full", "packaged", "quick"}),
    ("Docs    write Results .docx",                 "17_write_results_docx.py",     {},                              False, {"full", "packaged", "quick"}),
    ("Docs    write Supplement .docx",              "21_write_supplement.py",       {},                              False, {"full", "packaged", "quick"}),
]
NOVELTY = ["00_setup_and_fit.py", "01_dose_response.py", "02_shap_interactions.py",
           "03_component_level.py", "04_effect_modification.py", "05_mixture_score.py"]
RIGOR = ["00_setup_and_fit.py", "06_temporal_validation.py", "07_pr_curve.py", "08_shap_waterfall.py",
         "09_shap_stability.py", "10_correlation_heatmap.py", "11_wqs_mixture.py", "12_reporting_checklists.py"]


# ----------------------------------------------------------------------
def log(msg: str = "") -> None:
    print(msg, flush=True)
    OUT.mkdir(exist_ok=True)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(msg + "\n")


def venv_python() -> Path:
    return VENV / ("Scripts/python.exe" if os.name == "nt" else "bin/python")


def inside_venv() -> bool:
    return Path(sys.prefix).resolve() == VENV.resolve()


def pause_and_exit(code: int) -> None:
    # keep a double-clicked window open so the message can be read
    # (only the outer process pauses; the restarted copy inside .venv does not)
    if not os.environ.get("RUN_ME_CHILD") and sys.stdin and sys.stdin.isatty():
        try:
            input("\nPress Enter to close this window...")
        except EOFError:
            pass
    sys.exit(code)


def setup_environment() -> None:
    """First run: build .venv and install packages. Later runs: reuse it."""
    py = venv_python()
    marker = VENV / "INSTALL_COMPLETE"
    if not py.exists():
        print("Creating a private Python environment in .venv (first run only) ...")
        venv.EnvBuilder(with_pip=True, clear=False).create(VENV)
    if not marker.exists():
        print("Installing packages (first run only, about 5-15 minutes) ...")
        pip = [str(py), "-m", "pip", "install", "--disable-pip-version-check"]
        subprocess.call(pip + ["--upgrade", "pip"])
        req = CODE / "requirements.txt"
        pinned = [l.strip() for l in req.read_text().splitlines() if l.strip() and not l.startswith("#")]
        if subprocess.call(pip + pinned + EXTRA_PACKAGES) != 0:
            print("\nThe exact pinned versions could not be installed on this computer "
                  "(usually a newer Python version). Installing the closest available versions instead;\n"
                  "results may differ in the third decimal place.")
            loose = [p.split("==")[0] for p in pinned]
            if subprocess.call(pip + loose + EXTRA_PACKAGES) != 0:
                print("\nPackage installation failed. Check your internet connection and run again.")
                pause_and_exit(1)
        marker.write_text("ok\n")


def choose_mode(argv) -> tuple[str, bool]:
    mode, extras = None, True
    if "--mode" in argv:
        mode = argv[argv.index("--mode") + 1].lower()
    if "--no-extras" in argv:
        extras = False
    if mode in {"full", "packaged", "quick"}:
        return mode, extras
    print(__doc__.split("How to start it")[0].split("What it does, in order")[1])
    try:
        ans = input("Choose a run [1 = FULL, 2 = PACKAGED, 3 = QUICK; Enter = 2]: ").strip()
        ex = input("Also rerun the exploratory analyses (folders 07 and 08)? [Y/n]: ").strip().lower()
    except EOFError:
        ans, ex = "", ""
    mode = {"1": "full", "3": "quick"}.get(ans, "packaged")
    return mode, ex not in {"n", "no"}


def prepare_packaged_inputs() -> None:
    """PACKAGED/QUICK: put the shipped cleaned data where the scripts expect it,
    together with the three step-2 tables later steps read."""
    dp = ROOT / "data_processed"
    met = OUT / "metrics"
    dp.mkdir(exist_ok=True)
    met.mkdir(parents=True, exist_ok=True)
    shutil.copy2(ROOT / "05_Data_Processed" / "nhanes_with_outcome.csv", dp / "nhanes_with_outcome.csv")
    for name in ["exclusion_flow.csv", "missingness_report.csv",
                 "missingness_report_age_filtered_precohort.csv", "outcome_definition_comparison.csv"]:
        src = ROOT / "04_Results_Tables" / name
        if src.exists():
            shutil.copy2(src, met / name)
    log("Using the cleaned data shipped in 05_Data_Processed/ (steps 0-3 skipped).")


def run_script(path: Path, args, cwd: Path, env) -> bool:
    cmd = [sys.executable, str(path)] + list(args)
    with subprocess.Popen(cmd, cwd=str(cwd), env=env, stdout=subprocess.PIPE,
                          stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace") as p:
        with open(LOG, "a", encoding="utf-8") as f:
            for line in p.stdout:
                f.write(line)
                if line.strip():
                    sys.stdout.write("    " + line)
        return p.wait() == 0


def run_additions(folder: str, scripts, env) -> None:
    """The 07/08 scripts read 'SUBMISSION/05_Data_Processed/...' relative to the
    folder they run in and write their figures and tables there, so each runs in
    a scratch folder under outputs/ that mirrors that layout."""
    work = OUT / folder
    sub = work / "SUBMISSION"
    (sub / "05_Data_Processed").mkdir(parents=True, exist_ok=True)
    for f in ["model_ready.csv", "train_seqn.csv", "test_seqn.csv"]:
        shutil.copy2(ROOT / "data_processed" / f, sub / "05_Data_Processed" / f)
    shutil.copytree(CODE, sub / "06_Code", dirs_exist_ok=True)
    src = ROOT / folder / "code"
    for f in src.glob("*.py"):
        shutil.copy2(f, work / f.name)
    for s in scripts:
        if not (work / s).exists():
            log(f"  [skip] {folder}/code/{s} not found")
            if s == "00_setup_and_fit.py":
                return
            continue
        t0 = time.time()
        ok = run_script(work / s, [], work, env)
        log(f"  {'done ' if ok else 'FAILED'} {folder}/{s}  ({time.time() - t0:.0f} s)")
        if not ok and s == "00_setup_and_fit.py":
            log("  setup failed, skipping the rest of this folder")
            return


def main() -> None:
    if sys.version_info < MIN_PY:
        print(f"Python {MIN_PY[0]}.{MIN_PY[1]} or newer is needed; this is {sys.version.split()[0]}.\n"
              "Install it from https://www.python.org/downloads/ and run again.")
        pause_and_exit(1)

    if not inside_venv():
        setup_environment()
        # restart this same file inside the private environment
        code = subprocess.call([str(venv_python()), str(Path(__file__).resolve())] + sys.argv[1:],
                               env=dict(os.environ, RUN_ME_CHILD="1"))
        pause_and_exit(code)

    mode, extras = choose_mode(sys.argv)
    OUT.mkdir(exist_ok=True)
    LOG.write_text("", encoding="utf-8")
    env = dict(os.environ, PYTHONUTF8="1", PYTHONIOENCODING="utf-8", MPLBACKEND="Agg", RUN_ME_MODE=mode)
    log(f"RUN_ME started {time.strftime('%Y-%m-%d %H:%M')}  mode={mode.upper()}  extras={extras}")
    log(f"Python {sys.version.split()[0]} on {sys.platform}")

    if mode in {"packaged", "quick"}:
        prepare_packaged_inputs()

    t_all = time.time()
    for label, script, margs, required, modes in CORE_STEPS:
        if mode not in modes:
            continue
        path = CODE / script
        if not path.exists():
            log(f"[skip] {label}: {script} not found")
            continue
        log(f"\n=== {label}  ({script}) ===")
        t0 = time.time()
        ok = run_script(path, margs.get(mode, []), ROOT, env)
        log(f"--- {'done' if ok else 'FAILED'} in {time.time() - t0:.0f} s")
        if not ok and required:
            log(f"\nThe required step '{label}' failed. The last lines above show why; "
                f"the full log is in {LOG}.")
            pause_and_exit(1)
        if not ok:
            log("    (optional step; continuing)")

    if extras:
        log("\n=== Exploratory additions: 07_Novelty_Additions ===")
        run_additions("07_Novelty_Additions", NOVELTY, env)
        log("\n=== Exploratory additions: 08_Q1_Rigor_Additions ===")
        run_additions("08_Q1_Rigor_Additions", RIGOR, env)

    log("\n=== Reproduction check ===")
    run_script(CODE / "28_compare_to_reference.py", [], ROOT, env)
    log(f"\nAll done in {(time.time() - t_all) / 60:.1f} minutes.")
    log("Results: outputs/metrics (tables), outputs/figures and outputs/figures_pub (figures), "
        "manuscript/ (documents), outputs/reproduction_report.txt (comparison with the published numbers).")
    pause_and_exit(0)


if __name__ == "__main__":
    main()
