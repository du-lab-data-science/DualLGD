"""Compute evaluation metrics (optionally grouped by adduct type).

Usage examples:
    # Basic usage — overall metrics only
    python compute_metrics.py --pred_dir ./outputs/2025-10-30/07-51-48-dev/preds --pred_prefix dev

    # With adduct breakdown
    python compute_metrics.py --pred_dir ./outputs/2025-10-30/07-51-48-dev/preds --pred_prefix dev \
        --by_adduct --labels data/mona_plus/labels.tsv --test_split data/mona_plus/splits/test.tsv

    # Full k range (1..100) and MCES
    python compute_metrics.py --pred_dir ./outputs/2025-10-30/07-51-48-dev/preds --pred_prefix dev \
        --full --mces

    # Custom output prefix and number of parallel jobs
    python compute_metrics.py --pred_dir ./outputs/2025-10-30/07-51-48-dev/preds --pred_prefix dev \
        --output_prefix metrics_mona_plus --n_jobs 24
"""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
import os
import pickle
import random
import shutil
import time
from collections import Counter, defaultdict

import pandas as pd
import pulp
from myopic_mces import MCES
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, DataStructs
from tqdm import tqdm

RDLogger.DisableLog("rdApp.*")

# ---------------------------------------------------------------------------
# Tautomer canonicalization (rdkit version-aware)
# ---------------------------------------------------------------------------
try:
    from rdkit.Chem.MolStandardize.tautomer import (
        TautomerCanonicalizer,
        TautomerTransform,
    )

    _RD_TAUTOMER_CANONICALIZER = "v1"
    _TAUTOMER_TRANSFORMS = (
        TautomerTransform(
            "1,3 heteroatom H shift",
            "[#7,S,O,Se,Te;!H0]-[#7X2,#6,#15]=[#7,#16,#8,Se,Te]",
        ),
        TautomerTransform(
            "1,3 (thio)keto/enol r", "[O,S,Se,Te;X2!H0]-[C]=[C]"
        ),
    )
except ModuleNotFoundError:
    from rdkit.Chem.MolStandardize.rdMolStandardize import TautomerEnumerator

    _RD_TAUTOMER_CANONICALIZER = "v2"


# ---------------------------------------------------------------------------
# Molecule helpers
# ---------------------------------------------------------------------------
def canonical_mol_from_inchi(inchi):
    """Canonicalize mol after Chem.MolFromInchi (tautomer-aware)."""
    mol = Chem.MolFromInchi(inchi)
    if mol is None:
        return None
    try:
        if _RD_TAUTOMER_CANONICALIZER == "v1":
            canon = TautomerCanonicalizer(transforms=_TAUTOMER_TRANSFORMS)
            mol = canon.canonicalize(mol)
        else:
            mol = TautomerEnumerator().Canonicalize(mol)
    except (Chem.rdchem.MolSanitizeException, RuntimeError):
        return None
    return mol


def _copy_mol(mol):
    """Return a defensive RDKit mol copy, or None when copying fails."""
    if mol is None:
        return None
    try:
        return Chem.Mol(mol)
    except Exception:
        return None


def _sanitize_and_validate_inplace(mol):
    """Sanitize and validate a mol in-place.

    Returns True only for a single-fragment valid molecule.
    """
    if mol is None:
        return False
    try:
        Chem.SanitizeMol(mol)
    except (ValueError, Exception):
        return False
    if Chem.MolToSmiles(mol) is None:
        return False
    try:
        frags = Chem.rdmolops.GetMolFrags(mol, asMols=True, sanitizeFrags=True)
    except Exception:
        return False
    return len(frags) == 1


def is_valid(mol):
    # IMPORTANT: validate on a copy to avoid mutating original mol objects.
    mol = _copy_mol(mol)
    return _sanitize_and_validate_inplace(mol)


import re as _re

_BOND_DICT = [
    None,
    Chem.rdchem.BondType.SINGLE,
    Chem.rdchem.BondType.DOUBLE,
    Chem.rdchem.BondType.TRIPLE,
]


def _try_repair_mol(mol):
    """Attempt to repair an invalid mol and return a valid Mol, or None."""
    if mol is None:
        return None
    try:
        rw = Chem.RWMol(mol)
    except Exception:
        return None

    # Phase 1: clear aromatic flags and convert AROMATIC bonds to SINGLE.
    for atom in rw.GetAtoms():
        atom.SetIsAromatic(False)
    for bond in rw.GetBonds():
        if bond.GetBondType() == Chem.rdchem.BondType.AROMATIC:
            bond.SetBondType(Chem.rdchem.BondType.SINGLE)
        bond.SetIsAromatic(False)

    # Phase 2: iteratively fix over-valenced atoms by downgrading bonds.
    for _ in range(50):
        try:
            Chem.SanitizeMol(rw, sanitizeOps=Chem.SanitizeFlags.SANITIZE_PROPERTIES)
            break
        except ValueError as e:
            e_str = str(e)
            p = e_str.find("#")
            if p == -1:
                return None
            nums = list(map(int, _re.findall(r"\d+", e_str[p:])))
            if len(nums) < 2:
                return None
            idx = nums[0]
            bonds = []
            for b in rw.GetAtomWithIdx(idx).GetBonds():
                bonds.append(
                    (b.GetIdx(), int(b.GetBondType()), b.GetBeginAtomIdx(), b.GetEndAtomIdx())
                )
            bonds.sort(key=lambda t: t[1], reverse=True)
            if not bonds:
                return None
            _, btype, start, end = bonds[0]
            rw.RemoveBond(start, end)
            new_type = btype - 1
            if new_type >= 1 and new_type < len(_BOND_DICT):
                rw.AddBond(start, end, _BOND_DICT[new_type])
        except Exception:
            return None

    # Phase 3: full sanitization.
    try:
        Chem.SanitizeMol(rw)
    except Exception:
        return None

    mol_out = rw.GetMol()

    # Phase 4: keep largest fragment if multi-fragment.
    try:
        frags = Chem.rdmolops.GetMolFrags(mol_out, asMols=True, sanitizeFrags=True)
    except Exception:
        return None
    if len(frags) > 1:
        mol_out = max(frags, key=lambda m: m.GetNumAtoms())

    if Chem.MolToSmiles(mol_out) is None:
        return None
    return mol_out


# ---------------------------------------------------------------------------
# Per-sample metric computation (runs inside parallel workers)
# ---------------------------------------------------------------------------
def _compute_one(
    t_inchi,
    p_inchis,
    solver,
    do_mces,
    k_max,
    matching_method="inchi",
    mces_threshold=100.0,
    mces_solver_options=None,
):
    """Compute metrics for a single (true, preds) pair.

    matching_method controls how accuracy (exact match) is determined:
      - 'inchi':    raw InChI string comparison (default, same as notebook)
      - 'inchikey': first 14 chars of InChIKey (connectivity layer)
      - 'smiles':   canonical SMILES comparison
    """
    RDLogger.DisableLog("rdApp.*")

    true_mol = canonical_mol_from_inchi(t_inchi)
    if true_mol is None:
        raise ValueError(f"Invalid true InChI encountered during metric computation: {t_inchi}")

    true_smi = Chem.MolToSmiles(true_mol)
    true_fp = AllChem.GetMorganFingerprintAsBitVect(true_mol, 2, nBits=2048)
    true_num_bonds = true_mol.GetNumBonds()

    true_mces_fallback = float(true_num_bonds * 2)
    solver_options = dict(mces_solver_options or {})
    if "msg" not in solver_options:
        solver_options["msg"] = False

    # Prefix arrays for best-so-far metrics (index 0 = no predictions used)
    best_mces = [true_mces_fallback]
    best_tanimoto = [0.0]
    best_cosine = [0.0]

    for pi in p_inchis:
        pmol = canonical_mol_from_inchi(pi)

        if pmol is None:
            # Invalid predicted structure: assign pessimistic defaults.
            mces_val = true_mces_fallback
            tanimoto_val = 0.0
            cosine_val = 0.0
        else:
            pmol_num_bonds = pmol.GetNumBonds()

            # MCES / bond-sum fallback
            if do_mces:
                current_best_mces = float(best_mces[-1])
                try:
                    if current_best_mces <= 0.0:
                        # MCES cannot improve once we already reached 0.
                        mces_val = current_best_mces
                    else:
                        pmol_smi = Chem.MolToSmiles(pmol)

                        # Exact shortcut for identical canonicalized structures.
                        if pmol_smi == true_smi:
                            mces_val = 0.0
                        else:
                            # Safe lower bound on edge-edit distance.
                            bond_count_lb = float(abs(true_num_bonds - pmol_num_bonds))

                            # If lower bound is already not better than current best,
                            # we can skip expensive MCES entirely.
                            if bond_count_lb >= current_best_mces:
                                mces_val = bond_count_lb
                            else:
                                if float(mces_threshold) == -1.0:
                                    effective_threshold = -1.0
                                else:
                                    effective_threshold = min(
                                        float(mces_threshold), current_best_mces
                                    )

                                mces_val = float(
                                    MCES(
                                        true_smi,
                                        pmol_smi,
                                        solver=solver,
                                        threshold=effective_threshold,
                                        always_stronger_bound=False,
                                        solver_options=solver_options,
                                        catch_errors=True,
                                    )[1]
                                )
                                if mces_val < 0:
                                    mces_val = float(true_num_bonds + pmol_num_bonds)
                except Exception:
                    mces_val = float(true_num_bonds + pmol_num_bonds)
            else:
                mces_val = float(true_num_bonds + pmol_num_bonds)

            # Fingerprint similarities
            try:
                pmol_fp = AllChem.GetMorganFingerprintAsBitVect(pmol, 2, nBits=2048)
                tanimoto_val = DataStructs.TanimotoSimilarity(true_fp, pmol_fp)
                cosine_val = DataStructs.CosineSimilarity(true_fp, pmol_fp)
            except Exception:
                tanimoto_val = 0.0
                cosine_val = 0.0

        best_mces.append(min(best_mces[-1], mces_val))
        best_tanimoto.append(max(best_tanimoto[-1], tanimoto_val))
        best_cosine.append(max(best_cosine[-1], cosine_val))

    # Accuracy: exact-match check using the chosen matching method
    if matching_method == "inchi":
        true_key = t_inchi
        pred_keys = p_inchis
    elif matching_method == "inchikey":
        true_key = Chem.InchiToInchiKey(t_inchi)[:14]
        pred_keys = []
        for pi in p_inchis:
            try:
                pred_keys.append(Chem.InchiToInchiKey(pi)[:14])
            except Exception:
                pred_keys.append(None)
    elif matching_method == "smiles":
        true_key = true_smi
        pred_keys = []
        for pi in p_inchis:
            try:
                pred_keys.append(Chem.MolToSmiles(canonical_mol_from_inchi(pi)))
            except Exception:
                pred_keys.append(None)
    else:
        raise ValueError(f"Unknown matching_method: {matching_method}")

    try:
        earliest_idx = pred_keys.index(true_key)
    except ValueError:
        earliest_idx = -1

    m = {}
    for k in range(1, k_max + 1):
        idx = min(k, len(p_inchis))
        tan_k = best_tanimoto[idx]
        m[f"acc@{k}"] = 1.0 if (earliest_idx != -1 and earliest_idx < k) else 0.0
        m[f"mces@{k}"] = best_mces[idx]
        m[f"tanimoto@{k}"] = tan_k
        m[f"cosine@{k}"] = best_cosine[idx]
        m[f"close_match@{k}"] = 1.0 if tan_k >= 0.675 else 0.0
        m[f"meaningful_match@{k}"] = 1.0 if tan_k >= 0.4 else 0.0
    return m


# ---------------------------------------------------------------------------
# Preprocessing: mol lists → deduplicated InChI lists (serial, IO-bound)
# ---------------------------------------------------------------------------
def _preprocess(true_mols, pred_mols, k_max, repair=True):
    """Convert mol objects to InChI strings with deduplication by frequency.

    When *repair* is True, samples with zero valid candidates get a second pass
    where ``_try_repair_mol`` is applied to every candidate.  Samples that
    already have at least one valid candidate are left unchanged.
    """
    true_inchis = []
    pred_inchis = []
    n_repaired = 0
    n_samples_repaired = 0
    for i in range(len(true_mols)):
        # First pass: normal validation.
        local = []
        for j in range(len(pred_mols[i])):
            pred_mol = _copy_mol(pred_mols[i][j])
            if pred_mol is None:
                continue
            if _sanitize_and_validate_inplace(pred_mol):
                local.append(Chem.MolToInchi(pred_mol))

        # Second pass: repair only when the sample has no valid candidates.
        if not local and repair:
            for j in range(len(pred_mols[i])):
                pred_mol = _copy_mol(pred_mols[i][j])
                repaired = _try_repair_mol(pred_mol)
                if repaired is not None:
                    inchi = Chem.MolToInchi(repaired)
                    if inchi is not None:
                        local.append(inchi)
                        n_repaired += 1
            if local:
                n_samples_repaired += 1

        counts = Counter(local)
        local = [item for item, _ in counts.most_common()]
        # Keep at most k_max + 1 to cover all k values we evaluate
        pred_inchis.append(local[: k_max + 1])

        true_mol = _copy_mol(true_mols[i])
        if true_mol is None:
            raise ValueError(f"Invalid true molecule at index {i}: cannot copy RDKit mol")
        true_inchis.append(Chem.MolToInchi(true_mol))

    if n_repaired > 0:
        print(
            f"Repair: recovered {n_repaired} candidates across "
            f"{n_samples_repaired} previously-empty samples"
        )
    return true_inchis, pred_inchis


# ---------------------------------------------------------------------------
# Aggregation / writing helpers
# ---------------------------------------------------------------------------
def _aggregate_results(results):
    """Aggregate per-sample metric dicts into dataset-level averages."""
    valid_results = [r for r in results if isinstance(r, dict)]
    n = len(valid_results)
    if n == 0:
        raise ValueError("Cannot aggregate metrics for an empty result set")

    dropped = len(results) - n
    if dropped > 0:
        print(f"Warning: dropped {dropped} failed sample(s) from metric aggregation")

    final = defaultdict(float)
    for r in valid_results:
        for key, val in r.items():
            final[key] += val
    for key in final:
        final[key] /= n
    return final


def _write_metrics_csv(metrics, csv_path):
    """Persist aggregated metrics to CSV."""
    pd.DataFrame(metrics, index=[0]).to_csv(csv_path, index=False)
    print(f"  → Saved: {csv_path}")


def _metrics_to_json(metrics):
    if metrics is None:
        return ""
    normalized = {k: float(v) for k, v in metrics.items()}
    return json.dumps(normalized, ensure_ascii=False, sort_keys=True)


def _metrics_from_json(metrics_json):
    if not metrics_json or (isinstance(metrics_json, float) and pd.isna(metrics_json)):
        return None
    try:
        payload = json.loads(metrics_json)
    except Exception:
        return None
    if not isinstance(payload, dict):
        return None
    normalized = {}
    for key, val in payload.items():
        try:
            normalized[key] = float(val)
        except Exception:
            normalized[key] = val
    return normalized


def _build_mces_cache_key(
    t_inchi,
    p_inchis,
    *,
    k_max,
    matching_method,
    mces_threshold,
    solver_options,
):
    # Cache should be solver-insensitive so runs can reuse entries across
    # GUROBI / CBC / HiGHS. Keep only options that may affect numerical
    # optimality (e.g., time limit) in the key.
    quality_options = {}
    if solver_options and "timeLimit" in solver_options:
        try:
            quality_options["timeLimit"] = float(solver_options["timeLimit"])
        except Exception:
            quality_options["timeLimit"] = solver_options["timeLimit"]

    payload = {
        "schema_version": 2,
        "true_inchi": t_inchi,
        "pred_inchis": p_inchis,
        "k_max": int(k_max),
        "matching_method": str(matching_method),
        "mces_threshold": float(mces_threshold),
        "quality_options": quality_options,
    }
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _append_suffix_to_csv_path(csv_path, suffix):
    base, ext = os.path.splitext(csv_path)
    if ext.lower() != ".csv":
        return f"{csv_path}{suffix}.csv"
    return f"{base}{suffix}.csv"


_MCES_CACHE_COLUMNS = [
    "sample_idx",
    "cache_key",
    "status",
    "num_candidates",
    "mces_time_sec",
    "error_message",
    "computed_at",
    "metrics_json",
]


_MCES_SAMPLE_REPORT_COLUMNS = [
    "sample_idx",
    "status",
    "cache_hit",
    "num_candidates",
    "mces_time_sec",
    "cache_key",
    "error_message",
    "metrics_json",
]


def _resolve_max_workers(n_jobs):
    cpu_count = os.cpu_count() or 1
    if n_jobs is None:
        return 1
    n_jobs = int(n_jobs)
    if n_jobs == 0:
        raise ValueError("n_jobs must not be 0")
    if n_jobs == -1:
        return cpu_count
    if n_jobs < -1:
        return max(1, cpu_count + 1 + n_jobs)
    return n_jobs


def _init_csv_with_columns(csv_path, columns):
    if not csv_path:
        return
    out_dir = os.path.dirname(csv_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    pd.DataFrame(columns=list(columns)).to_csv(csv_path, index=False)


def _append_csv_rows(csv_path, rows, columns):
    if not csv_path or not rows:
        return
    out_dir = os.path.dirname(csv_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    frame = pd.DataFrame(rows).reindex(columns=list(columns))
    write_header = not os.path.exists(csv_path) or os.path.getsize(csv_path) == 0
    frame.to_csv(csv_path, mode="a", header=write_header, index=False)


def _normalize_existing_cache_for_append(cache_csv_path, existing_cache_df):
    if not cache_csv_path:
        return set()

    if existing_cache_df is None or existing_cache_df.empty:
        _init_csv_with_columns(cache_csv_path, _MCES_CACHE_COLUMNS)
        return set()

    normalized = existing_cache_df.copy()
    if "status" in normalized.columns:
        normalized = normalized[normalized["status"].astype(str).str.lower() == "success"]
    else:
        normalized["status"] = "success"

    for col in _MCES_CACHE_COLUMNS:
        if col not in normalized.columns:
            normalized[col] = ""

    normalized = normalized.reindex(columns=_MCES_CACHE_COLUMNS)
    normalized = normalized[normalized["cache_key"].notna()]
    normalized["cache_key"] = normalized["cache_key"].astype(str).str.strip()
    normalized = normalized[normalized["cache_key"] != ""]
    normalized = normalized.drop_duplicates(subset=["cache_key"], keep="last")
    normalized.to_csv(cache_csv_path, index=False)
    return set(normalized["cache_key"].tolist())


def _append_mces_sample_record(
    record,
    sample_stats_csv_path,
    failed_samples_csv_path,
    no_valid_candidates_csv_path,
):
    _append_csv_rows(
        sample_stats_csv_path,
        [record],
        _MCES_SAMPLE_REPORT_COLUMNS,
    )

    status = str(record.get("status", "")).strip().lower()
    if status == "failed":
        _append_csv_rows(
            failed_samples_csv_path,
            [record],
            _MCES_SAMPLE_REPORT_COLUMNS,
        )
    elif status == "no_valid_candidates":
        _append_csv_rows(
            no_valid_candidates_csv_path,
            [record],
            _MCES_SAMPLE_REPORT_COLUMNS,
        )


def _append_mces_cache_record(cache_csv_path, cache_record):
    _append_csv_rows(cache_csv_path, [cache_record], _MCES_CACHE_COLUMNS)


def _load_mces_cache(cache_csv_path):
    if not cache_csv_path or not os.path.exists(cache_csv_path):
        return {}, pd.DataFrame()

    try:
        cache_df = pd.read_csv(cache_csv_path)
    except Exception as e:
        print(f"Warning: failed to read MCES cache {cache_csv_path}: {e}")
        return {}, pd.DataFrame()

    cache_map = {}
    if cache_df.empty:
        print(f"Loaded MCES cache entries: 0 from {cache_csv_path}")
        return cache_map, cache_df

    for _, row in cache_df.iterrows():
        status = str(row.get("status", "")).strip().lower()
        if status != "success":
            continue

        cache_key = str(row.get("cache_key", "")).strip()
        if not cache_key:
            continue

        metrics = _metrics_from_json(row.get("metrics_json", ""))
        if metrics is None:
            continue

        try:
            mces_time_sec = float(row.get("mces_time_sec", 0.0))
        except Exception:
            mces_time_sec = 0.0

        cache_map[cache_key] = {
            "metrics": metrics,
            "mces_time_sec": mces_time_sec,
        }

    print(f"Loaded MCES cache entries: {len(cache_map)} from {cache_csv_path}")
    return cache_map, cache_df


def _compute_one_safe(*args, catch_errors=True, **kwargs):
    t_inchi = args[0]
    p_inchis = args[1]
    started = time.perf_counter()
    no_valid_candidates = len(p_inchis) == 0

    try:
        metrics = _compute_one(*args, **kwargs)
        status = "no_valid_candidates" if no_valid_candidates else "success"
        error_message = ""
    except Exception as e:
        if not catch_errors:
            raise
        metrics = None
        status = "failed"
        error_message = f"{type(e).__name__}: {e}"

    elapsed = time.perf_counter() - started
    return {
        "true_inchi": t_inchi,
        "status": status,
        "metrics": metrics,
        "mces_time_sec": elapsed,
        "num_candidates": len(p_inchis),
        "error_message": error_message,
    }


def _resolve_mces_solver(
    mces_solver="auto",
    mces_solver_path=None,
    mces_threads=None,
    mces_time_limit=None,
):
    """Resolve solver name and options for myopic_mces.MCES()."""
    solver_options = {"msg": False}

    if mces_threads is not None:
        solver_options["threads"] = int(mces_threads)
    if mces_time_limit is not None:
        solver_options["timeLimit"] = float(mces_time_limit)

    solver = (mces_solver or "auto").strip()
    if solver.lower() == "auto":
        if mces_solver_path:
            # Explicit executable path support with Gurobi uses GUROBI_CMD.
            solver = "GUROBI_CMD"
        else:
            available_solvers = pulp.listSolvers(onlyAvailable=True)
            if not available_solvers:
                raise RuntimeError(
                    "MCES computation requested (--mces), but no pulp solver is available. "
                    "Ensure Gurobi is installed/licensed or provide --mces_solver_path to gurobi_cl."
                )
            preferred_solvers = [
                "GUROBI",
                "GUROBI_CMD",
                "HiGHS",
                "HiGHS_CMD",
                "CPLEX_CMD",
                "COIN_CMD",
                "PULP_CBC_CMD",
            ]
            for candidate in preferred_solvers:
                if candidate in available_solvers:
                    solver = candidate
                    break
            if solver is None:
                solver = available_solvers[0]

    if mces_solver_path:
        solver_options["path"] = mces_solver_path

    if solver == "GUROBI" and "path" in solver_options:
        print(
            "Warning: GUROBI uses Python API and ignores --mces_solver_path; "
            "switching to GUROBI_CMD."
        )
        solver = "GUROBI_CMD"

    if solver == "PULP_CBC_CMD" and "path" in solver_options:
        print(
            "Warning: PULP_CBC_CMD does not support --mces_solver_path; "
            "switching to COIN_CMD."
        )
        solver = "COIN_CMD"

    if solver == "COIN_CMD" and "path" not in solver_options:
        # Try to auto-locate cbc binary for COIN_CMD.
        cbc_path = shutil.which("cbc")
        if cbc_path:
            solver_options["path"] = cbc_path

    try:
        solver_obj = pulp.getSolver(solver, **solver_options)
        if not solver_obj or not solver_obj.available():
            raise RuntimeError(
                f"Solver {solver!r} is not available with options {solver_options}."
            )
    except Exception as e:
        raise RuntimeError(
            "Failed to configure MCES solver. "
            f"solver={solver!r}, options={solver_options}, error={e}. "
            "If using Gurobi, verify license (e.g., gurobi.lic / GRB_LICENSE_FILE)."
        ) from e

    return solver, solver_options


def compute_per_sample_metrics(
    true_mols, pred_mols, *,
    do_mces=False,
    do_full=False,
    n_jobs=-1,
    matching_method="inchi",
    mces_threshold=100.0,
    mces_solver="auto",
    mces_solver_path=None,
    mces_threads=None,
    mces_time_limit=None,
    progress_label="metrics",
    cache_csv_path=None,
    sample_stats_csv_path=None,
    failed_samples_csv_path=None,
    no_valid_candidates_csv_path=None,
    repair=True,
):
    """Compute per-sample metrics.

    matching_method: 'inchi' (default, same as notebook), 'inchikey', or 'smiles'.
    """
    k_max = 100 if do_full else 10
    true_inchis, pred_inchis = _preprocess(true_mols, pred_mols, k_max, repair=repair)

    solver = None
    solver_options = None
    if do_mces:
        solver, solver_options = _resolve_mces_solver(
            mces_solver=mces_solver,
            mces_solver_path=mces_solver_path,
            mces_threads=mces_threads,
            mces_time_limit=mces_time_limit,
        )
        print(f"MCES solver: {solver}")
        print(f"MCES solver options: {solver_options}")

    cache_enabled = bool(do_mces and cache_csv_path)
    cache_map = {}
    known_cache_keys = set()
    if cache_enabled:
        cache_map, existing_cache_df = _load_mces_cache(cache_csv_path)
        _normalize_existing_cache_for_append(cache_csv_path, existing_cache_df)
        known_cache_keys = set(cache_map.keys())
        if sample_stats_csv_path is None:
            sample_stats_csv_path = _append_suffix_to_csv_path(
                cache_csv_path, "_sample_stats"
            )
        if failed_samples_csv_path is None:
            failed_samples_csv_path = _append_suffix_to_csv_path(
                cache_csv_path, "_failed_samples"
            )
        if no_valid_candidates_csv_path is None:
            no_valid_candidates_csv_path = _append_suffix_to_csv_path(
                cache_csv_path, "_no_valid_candidates"
            )

    if do_mces:
        _init_csv_with_columns(sample_stats_csv_path, _MCES_SAMPLE_REPORT_COLUMNS)
        _init_csv_with_columns(failed_samples_csv_path, _MCES_SAMPLE_REPORT_COLUMNS)
        _init_csv_with_columns(
            no_valid_candidates_csv_path,
            _MCES_SAMPLE_REPORT_COLUMNS,
        )

    results = [None] * len(true_inchis)
    sample_records = []
    pending_indices = []

    for i in range(len(true_inchis)):
        cache_key = ""
        if cache_enabled:
            cache_key = _build_mces_cache_key(
                true_inchis[i],
                pred_inchis[i],
                k_max=k_max,
                matching_method=matching_method,
                mces_threshold=mces_threshold,
                solver_options=solver_options,
            )
            cache_item = cache_map.get(cache_key)

            if cache_item is not None:
                results[i] = cache_item["metrics"]
                sample_record = {
                    "sample_idx": i,
                    "status": "success",
                    "cache_hit": True,
                    "num_candidates": len(pred_inchis[i]),
                    "mces_time_sec": float(cache_item.get("mces_time_sec", 0.0)),
                    "cache_key": cache_key,
                    "error_message": "",
                    "metrics_json": _metrics_to_json(results[i]),
                }
                sample_records.append(sample_record)
                if do_mces:
                    _append_mces_sample_record(
                        sample_record,
                        sample_stats_csv_path,
                        failed_samples_csv_path,
                        no_valid_candidates_csv_path,
                    )

                continue

        pending_indices.append((i, cache_key))

    if cache_enabled:
        print(
            f"MCES cache hit: {len(true_inchis) - len(pending_indices)} / {len(true_inchis)} samples"
        )

    if pending_indices:
        max_workers = _resolve_max_workers(n_jobs)
        with tqdm(total=len(pending_indices), desc=progress_label) as progress:
            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                future_map = {
                    executor.submit(
                        _compute_one_safe,
                        true_inchis[i],
                        pred_inchis[i],
                        solver,
                        do_mces,
                        k_max,
                        matching_method,
                        mces_threshold,
                        solver_options,
                        catch_errors=do_mces,
                    ): (i, cache_key)
                    for i, cache_key in pending_indices
                }

                for fut in as_completed(future_map):
                    i, cache_key = future_map[fut]
                    try:
                        item = fut.result()
                    except Exception as e:
                        if not do_mces:
                            raise
                        item = {
                            "status": "failed",
                            "metrics": None,
                            "mces_time_sec": 0.0,
                            "num_candidates": len(pred_inchis[i]),
                            "error_message": f"{type(e).__name__}: {e}",
                        }

                    results[i] = item["metrics"]
                    status = item["status"]
                    sample_record = {
                        "sample_idx": i,
                        "status": status,
                        "cache_hit": False,
                        "num_candidates": int(item["num_candidates"]),
                        "mces_time_sec": float(item["mces_time_sec"]),
                        "cache_key": cache_key,
                        "error_message": item["error_message"],
                        "metrics_json": _metrics_to_json(item["metrics"]),
                    }
                    sample_records.append(sample_record)

                    if do_mces:
                        _append_mces_sample_record(
                            sample_record,
                            sample_stats_csv_path,
                            failed_samples_csv_path,
                            no_valid_candidates_csv_path,
                        )

                    if (
                        cache_enabled
                        and status == "success"
                        and cache_key
                        and cache_key not in known_cache_keys
                    ):
                        cache_record = {
                            "sample_idx": i,
                            "cache_key": cache_key,
                            "status": "success",
                            "num_candidates": int(item["num_candidates"]),
                            "mces_time_sec": float(item["mces_time_sec"]),
                            "error_message": "",
                            "computed_at": pd.Timestamp.utcnow().isoformat(),
                            "metrics_json": _metrics_to_json(item["metrics"]),
                        }
                        _append_mces_cache_record(cache_csv_path, cache_record)
                        known_cache_keys.add(cache_key)

                    progress.update(1)
    else:
        print("All samples were loaded from cache; no MCES recomputation needed.")

    if do_mces:
        n_failed = sum(1 for r in sample_records if r.get("status") == "failed")
        n_no_valid = sum(
            1 for r in sample_records if r.get("status") == "no_valid_candidates"
        )
        n_hit = sum(1 for r in sample_records if r.get("cache_hit"))
        print(
            "MCES sample status summary: "
            f"cache_hit={n_hit}, failed={n_failed}, no_valid_candidates={n_no_valid}"
        )
        if sample_stats_csv_path:
            print(f"MCES sample report path: {sample_stats_csv_path}")
        if failed_samples_csv_path:
            print(f"MCES failed samples path: {failed_samples_csv_path}")
        if no_valid_candidates_csv_path:
            print(f"MCES no-valid-candidates path: {no_valid_candidates_csv_path}")
        if cache_enabled:
            print(f"MCES cache path: {cache_csv_path}")

    return results


# ---------------------------------------------------------------------------
# Main metric computation
# ---------------------------------------------------------------------------
def compute_metrics(
    true_mols, pred_mols, csv_path, *,
    do_mces=False,
    do_full=False,
    n_jobs=-1,
    matching_method="inchi",
    mces_threshold=100.0,
    mces_solver="auto",
    mces_solver_path=None,
    mces_threads=None,
    mces_time_limit=None,
    mces_cache_csv_path=None,
    mces_sample_stats_csv_path=None,
    mces_failed_csv_path=None,
    mces_no_valid_candidates_csv_path=None,
    repair=True,
):
    """Compute metrics and write to *csv_path*.

    matching_method: 'inchi' (default, same as notebook), 'inchikey', or 'smiles'.
    """
    results = compute_per_sample_metrics(
        true_mols,
        pred_mols,
        do_mces=do_mces,
        do_full=do_full,
        n_jobs=n_jobs,
        matching_method=matching_method,
        mces_threshold=mces_threshold,
        mces_solver=mces_solver,
        mces_solver_path=mces_solver_path,
        mces_threads=mces_threads,
        mces_time_limit=mces_time_limit,
        progress_label=os.path.basename(csv_path),
        cache_csv_path=mces_cache_csv_path,
        sample_stats_csv_path=mces_sample_stats_csv_path,
        failed_samples_csv_path=mces_failed_csv_path,
        no_valid_candidates_csv_path=mces_no_valid_candidates_csv_path,
        repair=repair,
    )
    final = _aggregate_results(results)
    _write_metrics_csv(final, csv_path)
    return final


def _format_summary(metrics, label, do_mces=False):
    """Return a formatted summary string for key top-k metrics."""
    lines = [f"  [{label}]"]
    ks = [1, 5, 10]
    # Accuracy
    acc_vals = [f"top-{k}: {metrics.get(f'acc@{k}', 0):.4f}" for k in ks if f"acc@{k}" in metrics]
    if acc_vals:
        lines.append(f"    Accuracy       {', '.join(acc_vals)}")
    # Tanimoto
    tan_vals = [f"top-{k}: {metrics.get(f'tanimoto@{k}', 0):.4f}" for k in ks if f"tanimoto@{k}" in metrics]
    if tan_vals:
        lines.append(f"    Tanimoto       {', '.join(tan_vals)}")
    # Cosine
    cos_vals = [f"top-{k}: {metrics.get(f'cosine@{k}', 0):.4f}" for k in ks if f"cosine@{k}" in metrics]
    if cos_vals:
        lines.append(f"    Cosine         {', '.join(cos_vals)}")
    # MCES
    if do_mces:
        mces_vals = [f"top-{k}: {metrics.get(f'mces@{k}', 0):.4f}" for k in ks if f"mces@{k}" in metrics]
        if mces_vals:
            lines.append(f"    MCES           {', '.join(mces_vals)}")
    # Close / meaningful match
    close_vals = [f"top-{k}: {metrics.get(f'close_match@{k}', 0):.4f}" for k in ks if f"close_match@{k}" in metrics]
    if close_vals:
        lines.append(f"    Close match    {', '.join(close_vals)}")
    mean_vals = [f"top-{k}: {metrics.get(f'meaningful_match@{k}', 0):.4f}" for k in ks if f"meaningful_match@{k}" in metrics]
    if mean_vals:
        lines.append(f"    Meaningful     {', '.join(mean_vals)}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Adduct-grouped evaluation
# ---------------------------------------------------------------------------
def _load_test_metadata(labels_path, test_split_path):
    """Load test set metadata (adduct, SMILES) in the shuffled order used by
    the data loader, returning a DataFrame indexed by shuffled position."""
    labels_df = pd.read_csv(labels_path, sep="\t")
    full_spec_names = labels_df["spec"].astype(str).tolist()

    test_df = pd.read_csv(test_split_path, sep="\t")
    test_set = set(test_df["name"].astype(str).tolist())

    test_specs = [s for s in full_spec_names if s in test_set]

    random.seed(42)
    random.shuffle(test_specs)

    spec_to_row = {
        str(row["spec"]): row for _, row in labels_df.iterrows()
    }

    records = []
    for spec in test_specs:
        row = spec_to_row.get(spec, {})
        records.append(
            {
                "spec": spec,
                "ionization": row.get("ionization", "Unknown"),
                "smiles": row.get("smiles", "Unknown"),
            }
        )
    meta = pd.DataFrame(records)
    print(f"Test samples: {len(meta)}")
    print("Adduct distribution:")
    for adduct, cnt in sorted(meta["ionization"].value_counts().items()):
        print(f"  {adduct}: {cnt}")
    return meta


def _sanitize_adduct_name(adduct):
    return (
        adduct.replace("+", "plus")
        .replace("-", "minus")
        .replace("[", "")
        .replace("]", "")
    )


def compute_metrics_by_adduct(
    true_mols,
    pred_mols,
    csv_prefix,
    *,
    labels_path,
    test_split_path,
    do_mces=False,
    do_full=False,
    n_jobs=-1,
    matching_method="inchi",
    mces_threshold=100.0,
    mces_solver="auto",
    mces_solver_path=None,
    mces_threads=None,
    mces_time_limit=None,
    mces_cache_csv_path=None,
    mces_sample_stats_csv_path=None,
    mces_failed_csv_path=None,
    mces_no_valid_candidates_csv_path=None,
    verify=False,
    repair=True,
):
    """Compute overall metrics and per-adduct breakdowns."""
    meta = _load_test_metadata(labels_path, test_split_path)

    # Optional: verify true mols match the expected SMILES
    if verify:
        for i in range(min(len(true_mols), len(meta))):
            if not is_valid(true_mols[i]):
                continue
            t_inchi = Chem.MolToInchi(true_mols[i])
            expected_mol = Chem.MolFromSmiles(meta.iloc[i]["smiles"])
            if expected_mol is None:
                continue
            e_inchi = Chem.MolToInchi(expected_mol)
            if t_inchi != e_inchi:
                print(
                    f"Warning: InChI mismatch at index {i}: "
                    f"{t_inchi} vs {e_inchi} ({meta.iloc[i]['smiles']})"
                )

    # Per-adduct groups
    adduct_groups = defaultdict(list)
    for i in range(min(len(true_mols), len(meta))):
        adduct_groups[meta.iloc[i]["ionization"]].append(i)

    print(f"\nAdduct groups ({len(adduct_groups)}):")
    for adduct, indices in sorted(adduct_groups.items()):
        print(f"  {adduct}: {len(indices)} samples")

    # Collect all summaries
    summaries = []

    print("\nComputing per-sample metrics once for all samples...")
    all_results = compute_per_sample_metrics(
        true_mols,
        pred_mols,
        do_mces=do_mces,
        do_full=do_full,
        n_jobs=n_jobs,
        matching_method=matching_method,
        mces_threshold=mces_threshold,
        mces_solver=mces_solver,
        mces_solver_path=mces_solver_path,
        mces_threads=mces_threads,
        mces_time_limit=mces_time_limit,
        progress_label="per_sample_metrics",
        cache_csv_path=mces_cache_csv_path,
        sample_stats_csv_path=mces_sample_stats_csv_path,
        failed_samples_csv_path=mces_failed_csv_path,
        no_valid_candidates_csv_path=mces_no_valid_candidates_csv_path,
        repair=repair,
    )

    # Overall
    print("\nAggregating overall metrics...")
    overall = _aggregate_results(all_results)
    _write_metrics_csv(overall, f"{csv_prefix}_overall.csv")
    summaries.append(_format_summary(overall, f"Overall ({len(true_mols)} samples)", do_mces))

    # Per-adduct
    for adduct, indices in sorted(adduct_groups.items()):
        print(f"\nAggregating metrics for {adduct} ({len(indices)} samples)...")
        path = f"{csv_prefix}_{_sanitize_adduct_name(adduct)}.csv"
        adduct_metrics = _aggregate_results([all_results[i] for i in indices])
        _write_metrics_csv(adduct_metrics, path)
        summaries.append(_format_summary(adduct_metrics, f"{adduct} ({len(indices)} samples)", do_mces))

    # Print and write summary
    summary_text = _build_summary_header(matching_method, len(true_mols), adduct_groups, do_mces)
    summary_text += "\n".join(summaries)
    print(f"\n{'='*60}\n{summary_text}\n{'='*60}")

    summary_path = f"{csv_prefix}_summary.txt"
    with open(summary_path, "w") as f:
        f.write(summary_text + "\n")
    print(f"Summary saved to {summary_path}")


def _build_summary_header(matching_method, n_total, adduct_groups, do_mces):
    """Build the header section of the summary."""
    lines = [
        "EVALUATION SUMMARY",
        f"  Matching method: {matching_method}",
        f"  Total samples:   {n_total}",
        f"  Adduct groups:   {len(adduct_groups)}",
        f"  MCES computed:   {'yes' if do_mces else 'no'}",
        "",
    ]
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_predictions(pred_dir, pred_prefix, expected_samples=None):
    """Load true/pred pickle pairs produced by the test step.

    Supports naming conventions with and without the "_resume" token:
      1. Multi-rank:  {prefix}_rank_{r}_pred_{i}.pkl
                      {prefix}_resume_rank_{r}_pred_{i}.pkl
      2. Single-rank: {prefix}_pred_{i}.pkl
                      {prefix}_resume_pred_{i}.pkl

    Files are loaded sorted by (rank, index) to ensure deterministic ordering.

    Args:
        expected_samples: If given, truncate to this many samples to remove
            DistributedSampler padding duplicates that appear at the tail.
    """
    import glob as _glob
    import re

    # Discover all pred files and parse their (rank, index) keys
    pattern_multi = [
        os.path.join(pred_dir, f"{pred_prefix}_rank_*_pred_*.pkl"),
        os.path.join(pred_dir, f"{pred_prefix}_resume_rank_*_pred_*.pkl"),
    ]
    pattern_single = [
        os.path.join(pred_dir, f"{pred_prefix}_pred_*.pkl"),
        os.path.join(pred_dir, f"{pred_prefix}_resume_pred_*.pkl"),
    ]

    multi_files = sorted(set().union(*(_glob.glob(p) for p in pattern_multi)))
    single_files = sorted(set().union(*(_glob.glob(p) for p in pattern_single)))

    if multi_files:
        true_all, pred_all = [], []
        loaded_keys = []
        seen_keys = set()
        re_multi = re.compile(
            rf"{re.escape(pred_prefix)}(?:_resume)?_rank_(\d+)_pred_(\d+)\.pkl$"
        )

        for pred_path in multi_files:
            m = re_multi.search(os.path.basename(pred_path))
            if not m:
                continue

            rank, idx = int(m.group(1)), int(m.group(2))
            key = (rank, idx)
            if key in seen_keys:
                continue

            true_path = pred_path.replace("_pred_", "_true_")
            if not os.path.exists(true_path):
                print(f"Warning: missing true file {true_path}, skipping")
                continue

            with open(true_path, "rb") as f:
                true_all.extend(pickle.load(f))
            with open(pred_path, "rb") as f:
                pred_all.extend(pickle.load(f))

            seen_keys.add(key)
            loaded_keys.append(key)

        loaded_keys = sorted(loaded_keys)
        n_ranks = len({k[0] for k in loaded_keys})
        print(
            f"Loaded {len(true_all)} samples from {len(loaded_keys)} file pairs "
            f"({n_ranks} ranks) in {pred_dir}"
        )
    elif single_files:
        true_all, pred_all = [], []
        loaded_indices = []
        seen_indices = set()
        re_single = re.compile(
            rf"{re.escape(pred_prefix)}(?:_resume)?_pred_(\d+)\.pkl$"
        )

        for pred_path in single_files:
            m = re_single.search(os.path.basename(pred_path))
            if not m:
                continue

            idx = int(m.group(1))
            if idx in seen_indices:
                continue

            true_path = pred_path.replace("_pred_", "_true_")
            if not os.path.exists(true_path):
                print(f"Warning: missing true file {true_path}, skipping")
                continue

            with open(true_path, "rb") as f:
                true_all.extend(pickle.load(f))
            with open(pred_path, "rb") as f:
                pred_all.extend(pickle.load(f))

            seen_indices.add(idx)
            loaded_indices.append(idx)

        print(f"Loaded {len(true_all)} samples from {len(loaded_indices)} file pairs in {pred_dir}")
    else:
        print(f"No prediction files found in {pred_dir} with prefix '{pred_prefix}'")
        print(
            "Expected filename patterns include: "
            f"{pred_prefix}_rank_*_pred_*.pkl, "
            f"{pred_prefix}_resume_rank_*_pred_*.pkl, "
            f"{pred_prefix}_pred_*.pkl, "
            f"{pred_prefix}_resume_pred_*.pkl"
        )
        true_all, pred_all = [], []

    # DDP's DistributedSampler pads the dataset to make it evenly divisible
    # across ranks.  The padding duplicates appear at the tail of the last
    # rank's data.  Truncate them if the caller knows the true sample count.
    if expected_samples is not None and len(true_all) > expected_samples:
        n_removed = len(true_all) - expected_samples
        print(f"Truncating {n_removed} padding duplicate(s) "
              f"({len(true_all)} -> {expected_samples})")
        true_all = true_all[:expected_samples]
        pred_all = pred_all[:expected_samples]

    return true_all, pred_all


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(
        description="Compute evaluation metrics for structure prediction.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--pred_dir", required=True, help="Directory containing prediction pkl files")
    p.add_argument("--pred_prefix", required=True, help="Filename prefix (e.g. 'dev')")
    p.add_argument("--output_dir", required=True, help="Directory to write result CSVs and summary")
    p.add_argument("--output_prefix", default="metrics", help="Filename prefix for output CSVs (default: metrics)")
    p.add_argument("--mces", action="store_true", help="Compute MCES distance (slow)")
    p.add_argument("--full", action="store_true", help="Evaluate k=1..100 instead of k=1..10")
    p.add_argument("--n_jobs", type=int, default=-1, help="Number of parallel jobs (-1 = all CPUs)")
    p.add_argument("--by_adduct", action="store_true", help="Also compute per-adduct metrics")
    p.add_argument("--labels", default="data/mona_plus/labels.tsv", help="Labels TSV path")
    p.add_argument("--test_split", default="data/mona_plus/splits/test.tsv", help="Test split TSV path")
    p.add_argument("--matching_method", default="inchi", choices=["inchi", "inchikey", "smiles"],
                   help="Matching method for accuracy (default: inchi, same as notebook)")
    p.add_argument(
        "--mces_threshold",
        type=float,
        default=100.0,
        help=(
            "MCES threshold for exact solving (default: 100). "
            "Use -1 for always-exact (slowest)."
        ),
    )
    p.add_argument(
        "--mces_solver",
        default="auto",
        help=(
            "PuLP solver name for MCES (default: auto). "
            "Examples: GUROBI, GUROBI_CMD, auto, COIN_CMD, PULP_CBC_CMD."
        ),
    )
    p.add_argument(
        "--mces_solver_path",
        default=None,
        help=(
            "Path to solver executable (e.g. /path/to/gurobi_cl or /path/to/cbc). "
            "Useful when solver is installed but not in PATH."
        ),
    )
    p.add_argument(
        "--mces_threads",
        type=int,
        default=None,
        help="Threads per MCES solver instance (recommended: 1 with joblib parallelism).",
    )
    p.add_argument(
        "--mces_time_limit",
        type=float,
        default=None,
        help="Per-instance MILP time limit in seconds (solver dependent).",
    )
    p.add_argument("--verify", action="store_true", help="Verify true mols match labels SMILES")
    p.add_argument("--repair", action="store_true", default=True,
                   help="Attempt to repair invalid candidates before discarding (default: on)")
    p.add_argument("--no-repair", dest="repair", action="store_false",
                   help="Disable candidate repair")
    return p.parse_args()


def main():
    args = parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    csv_prefix = os.path.join(args.output_dir, args.output_prefix)
    mces_cache_csv_path = f"{csv_prefix}_mces_cache.csv"
    mces_sample_stats_csv_path = f"{csv_prefix}_mces_sample_stats.csv"
    mces_failed_csv_path = f"{csv_prefix}_mces_failed_samples.csv"
    mces_no_valid_candidates_csv_path = f"{csv_prefix}_mces_no_valid_candidates.csv"

    # Determine the expected number of test samples from the split file
    # so we can strip DistributedSampler padding duplicates.
    expected_samples = None
    if args.test_split and os.path.exists(args.test_split):
        expected_samples = len(pd.read_csv(args.test_split, sep="\t"))

    true_mols, pred_mols = load_predictions(
        args.pred_dir, args.pred_prefix, expected_samples=expected_samples
    )
    if len(true_mols) == 0:
        print("Error: no samples loaded. Check --pred_dir and --pred_prefix.")
        return

    print(f"Using matching method: {args.matching_method}")

    if args.by_adduct:
        compute_metrics_by_adduct(
            true_mols,
            pred_mols,
            csv_prefix,
            labels_path=args.labels,
            test_split_path=args.test_split,
            do_mces=args.mces,
            do_full=args.full,
            n_jobs=args.n_jobs,
            matching_method=args.matching_method,
            mces_threshold=args.mces_threshold,
            mces_solver=args.mces_solver,
            mces_solver_path=args.mces_solver_path,
            mces_threads=args.mces_threads,
            mces_time_limit=args.mces_time_limit,
            mces_cache_csv_path=mces_cache_csv_path,
            mces_sample_stats_csv_path=mces_sample_stats_csv_path,
            mces_failed_csv_path=mces_failed_csv_path,
            mces_no_valid_candidates_csv_path=mces_no_valid_candidates_csv_path,
            verify=args.verify,
            repair=args.repair,
        )
    else:
        overall = compute_metrics(
            true_mols,
            pred_mols,
            f"{csv_prefix}_overall.csv",
            do_mces=args.mces,
            do_full=args.full,
            n_jobs=args.n_jobs,
            matching_method=args.matching_method,
            mces_threshold=args.mces_threshold,
            mces_solver=args.mces_solver,
            mces_solver_path=args.mces_solver_path,
            mces_threads=args.mces_threads,
            mces_time_limit=args.mces_time_limit,
            mces_cache_csv_path=mces_cache_csv_path,
            mces_sample_stats_csv_path=mces_sample_stats_csv_path,
            mces_failed_csv_path=mces_failed_csv_path,
            mces_no_valid_candidates_csv_path=mces_no_valid_candidates_csv_path,
            repair=args.repair,
        )
        # Print and write summary for non-adduct mode too
        summary = _format_summary(overall, f"Overall ({len(true_mols)} samples)", args.mces)
        header = (
            f"EVALUATION SUMMARY\n"
            f"  Matching method: {args.matching_method}\n"
            f"  Total samples:   {len(true_mols)}\n"
            f"  MCES computed:   {'yes' if args.mces else 'no'}\n\n"
        )
        summary_text = header + summary
        print(f"\n{'='*60}\n{summary_text}\n{'='*60}")
        summary_path = f"{csv_prefix}_summary.txt"
        with open(summary_path, "w") as f:
            f.write(summary_text + "\n")
        print(f"Summary saved to {summary_path}")


if __name__ == "__main__":
    main()
