"""GO / MODIFY / ABANDON decision logic for the Phase-1 checkpoint.

IMPORTANT CONTRACT
------------------
This module does NOT decide anything a priori. It reads the metrics that were
actually MEASURED by a run and maps them to a suggested verdict via transparent,
editable criteria. Every threshold below is either (a) a signal sanity check
(e.g. "is there any positive correlation at all?") or (b) taken directly from the
REFERENCE POINTS the user supplied -- learned monocular height methods on
DFC2019-class data are in a rough 2-4 m RMSE band, and the shadow reference was
~3.84 m RMSE. Those are CONTEXT, not guaranteed targets, and are exposed as
parameters here so a human can adjust them. The final call is the human's; this
produces an auditable recommendation, not a fabricated result.

The verdict is driven by ONE central question from the spec: does the small
learned fusion head (C) beat trivial global-affine calibration (B) *on a fully
held-out city*? Plus guardrails: does the frozen depth prior carry any monotonic
height signal at all (A), and is the cross-city generalization gap catastrophic?
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class DecisionThresholds:
    # --- signal sanity (heuristics, editable) ---
    min_pearson_signal: float = 0.20      # below this, depth carries ~no monotone signal
    min_learned_gain: float = 0.10        # C must cut B's cross-city MAE by >=10% to "win"
    max_gen_gap_mult: float = 2.00        # cross-city MAE > 2x in-domain MAE = catastrophic
    # --- absolute usefulness band (USER-SUPPLIED reference points, context only) ---
    ref_rmse_good: float = 4.00           # upper edge of the published 2-4 m band
    ref_rmse_strong: float = 2.00         # lower edge = clearly competitive
    shadow_ref_rmse: float = 3.84         # researched shadow-based reference


def _get(agg: dict, key: str):
    v = agg.get(key) if isinstance(agg, dict) else None
    return v if (v is not None and v == v) else None  # drop None/NaN


def _crit(name, passed, measured, threshold, note):
    return {"name": name, "passed": bool(passed), "measured": measured,
            "threshold": threshold, "note": note}


def decide(results: dict, thr: DecisionThresholds | None = None,
           evidence_valid: bool = True) -> dict:
    """`results` is the merged metrics dict produced by scripts/03_make_report.

    Expected structure (missing pieces are tolerated -> INCONCLUSIVE):
      results['xcity'][name]['aggregate']['all'|'building']  # cross-city (TEST city)
      results['indomain'][name]['aggregate']['all']          # held-out TRAIN-city tiles
    where name in {A_raw_depth, B_global_affine, C_learned_fusion, ...}.
    """
    thr = thr or DecisionThresholds()
    crits: list[dict] = []
    notes: list[str] = []

    if not evidence_valid:
        return {
            "verdict": "INCONCLUSIVE",
            "summary": "Run used SYNTHETIC smoke-test data (or no real dataset). "
                       "Numbers exercise the pipeline only and are NOT feasibility "
                       "evidence. Re-run on real DFC2019 city-held-out data to decide.",
            "criteria": [], "human_note": HUMAN_NOTE,
        }

    xc = results.get("xcity", {})
    idn = results.get("indomain", {})

    A_x = xc.get("A_raw_depth", {}).get("aggregate", {}).get("all", {})
    A_id = idn.get("A_raw_depth", {}).get("aggregate", {}).get("all", {})
    B_x = xc.get("B_global_affine", {}).get("aggregate", {}).get("all", {})
    C_x = xc.get("C_learned_fusion", {}).get("aggregate", {}).get("all", {})
    C_id = idn.get("C_learned_fusion", {}).get("aggregate", {}).get("all", {})
    C_xb = xc.get("C_learned_fusion", {}).get("aggregate", {}).get("building", {})

    # ---- C1: does the frozen depth prior carry ANY monotone height signal? ----
    a_r = _get(A_id, "pearson_mean")
    if a_r is None:
        a_r = _get(A_x, "pearson_mean")
    c1_have = a_r is not None
    c1 = c1_have and a_r >= thr.min_pearson_signal
    crits.append(_crit("depth_prior_has_signal", c1, a_r, thr.min_pearson_signal,
                       "Baseline A Pearson r vs nDSM. Near-zero => no monotone height "
                       "signal; calibration/learning cannot rescue it."))

    # ---- C2: does learned fusion beat global affine ON THE HELD-OUT CITY? ----
    b_mae = _get(B_x, "mae_pooled")
    c_mae = _get(C_x, "mae_pooled")
    have_bc = (b_mae is not None) and (c_mae is not None)
    gain = ((b_mae - c_mae) / b_mae) if (have_bc and b_mae > 0) else None
    c2 = have_bc and gain is not None and gain >= thr.min_learned_gain
    crits.append(_crit("learned_beats_affine_xcity", c2, gain, thr.min_learned_gain,
                       "Relative cross-city MAE reduction of C over B. THE central test: "
                       "if <=0, the learned head adds nothing over trivial calibration."))

    # ---- C3: sanity -- did C learn in-domain at all? ----
    c_id_mae = _get(C_id, "mae_pooled")
    b_id_mae = _get(idn.get("B_global_affine", {}).get("aggregate", {}).get("all", {}),
                    "mae_pooled")
    c3 = (c_id_mae is not None and b_id_mae is not None and c_id_mae <= b_id_mae)
    crits.append(_crit("learned_ok_indomain", c3, c_id_mae, b_id_mae,
                       "C in-domain MAE <= B in-domain MAE (did the head fit at all?)."))

    # ---- C4: cross-city RMSE within the published reference band (CONTEXT) ----
    c_rmse = _get(C_x, "rmse_pooled")
    c4 = c_rmse is not None and c_rmse <= thr.ref_rmse_good
    strong = c_rmse is not None and c_rmse <= thr.ref_rmse_strong
    crits.append(_crit("xcity_rmse_in_reference_band", c4, c_rmse, thr.ref_rmse_good,
                       f"Cross-city RMSE vs the 2-4 m reference band (shadow ref "
                       f"~{thr.shadow_ref_rmse} m). CONTEXT, not a hard target."))

    # ---- C5: is the cross-city generalization gap catastrophic? ----
    gap_ok = (c_id_mae is not None and c_mae is not None and c_id_mae > 0
              and c_mae <= thr.max_gen_gap_mult * c_id_mae)
    crits.append(_crit("gen_gap_not_catastrophic", gap_ok,
                       (c_mae / c_id_mae) if (c_id_mae and c_mae) else None,
                       thr.max_gen_gap_mult,
                       "Cross-city MAE / in-domain MAE. Large => overfits the train city."))

    # ---- C6: is there building-class signal cross-city? ----
    cb_r = _get(C_xb, "pearson_mean")
    c6 = cb_r is not None and cb_r >= thr.min_pearson_signal
    crits.append(_crit("building_signal_xcity", c6, cb_r, thr.min_pearson_signal,
                       "C cross-city building-only Pearson r (buildings are the hard class)."))

    # ---------------- verdict mapping (transparent) ----------------
    have_core = c1_have and have_bc
    if not have_core:
        verdict = "INCONCLUSIVE"
        summary = ("Missing baselines (need A for signal + B and C for the head-vs-"
                   "calibration test). Likely torch/data absent. Re-run full A/B/C.")
    elif not c1:
        verdict = "ABANDON"
        summary = (f"Frozen depth prior shows ~no monotone height signal (A r={a_r:.3f} "
                   f"< {thr.min_pearson_signal}). The core premise fails on this data; "
                   "do not scale this architecture. Reconsider the input modality/prior.")
    elif c2 and (c4 or c6) and gap_ok:
        tag = "strong" if strong else "within reference band"
        verdict = "GO"
        summary = (f"Learned fusion beats affine cross-city by {gain*100:.0f}% "
                   f"(MAE {c_mae:.2f} vs {b_mae:.2f} m), cross-city RMSE {c_rmse:.2f} m "
                   f"({tag}), generalization gap acceptable. Proceed to build the full "
                   "pipeline around this swappable height head.")
    elif c1 and not c2:
        verdict = "MODIFY"
        summary = ("Depth prior carries signal, but the small learned head does NOT beat "
                   "trivial global-affine calibration cross-city "
                   f"(C MAE {c_mae}, B MAE {b_mae}). Do not build the heavy learned system "
                   "as-is: try richer features / domain adaptation / a stronger head, or "
                   "ship affine calibration as the honest baseline. Re-test before scaling.")
    else:
        verdict = "MODIFY"
        summary = ("Mixed signal: learned head shows some benefit but fails a guardrail "
                   "(reference band, building signal, or generalization gap). Iterate on "
                   "the weak axis and re-evaluate cross-city before committing.")

    return {"verdict": verdict, "summary": summary, "criteria": crits,
            "human_note": HUMAN_NOTE}


HUMAN_NOTE = (
    "These criteria are decision AIDS derived from signal sanity checks and the "
    "user-supplied reference points (2-4 m RMSE band; ~3.84 m shadow reference). "
    "They are editable (DecisionThresholds) and are NOT authoritative pass/fail "
    "targets. A human reviewer makes the final GO/MODIFY/ABANDON call using these "
    "measured numbers plus qualitative inspection of the error maps."
)
