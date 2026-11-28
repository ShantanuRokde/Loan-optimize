"""Agent 4 – Risk Discovery & Machine Learning Agent.

For the selected cohort: builds the dynamic target, splits chronologically
(earlier mature quarters -> train, latest mature quarter -> held-out), trains a
DecisionTreeClassifier (+ optional RandomForest challenger), evaluates it,
extracts one customer segment per leaf, computes IV/WoE, scores every loan and
registers the model. Retraining is skipped when a model with the same
signature (data, cohort, quarters, parameters) is already registered.
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd

from agents.base import AgentError, AgentResult, BaseAgent, load_report, save_report
from agents.data_agent import current_data_hash, get_processed_datasets
from analytics.cohort_calculation import cohort_definition, cohort_quarter_table, loan_cohort_frame
from backend.config import get_config
from database import repository as repo
from models import model_registry as reg
from models.decision_tree import TrainedModels, build_modeling_data, leaf_default_rates, score, train_models
from models.feature_engineering import FeatureSpec
from models.information_value import information_values
from models.model_evaluation import (
    aggregate_importance,
    calibration_table,
    classification_metrics,
    permutation_importance_by_feature,
)
from models.segmentation import assign_segments, extract_tree, segment_profiles
from utils.helpers import utcnow

REPORT_NAME = "risk_discovery"
IV_FEATURES = ["credit_score", "enquiries_6m", "existing_loans", "age", "income_band", "customer_segment",
               "occupation", "loan_type", "region", "requested_amount", "loan_amount", "interest_rate",
               "loan_tenure_months"]
SCORE_COLUMNS = ["loan_account_no", "disbursal_quarter", "disbursal_date", "split", "target", "breach_mob",
                 "segment_id", "leaf_node", "risk_score", "probability_of_default", "predicted_class",
                 "challenger_score", "credit_score", "enquiries_6m", "existing_loans", "age", "income_band",
                 "customer_segment", "occupation", "loan_type", "region", "branch", "requested_amount",
                 "loan_amount", "interest_rate", "loan_tenure_months"]


def resolve_cohort(ctx: dict, requested: str | None) -> str:
    if requested:
        return requested.strip().upper()
    if ctx.get("selected_cohort"):
        return ctx["selected_cohort"]
    rep = load_report("cohort_analysis")
    if rep and rep.get("selection", {}).get("cohort"):
        return rep["selection"]["cohort"]
    cfg_c = get_config().cohort.selected_cohort
    if cfg_c:
        return cfg_c
    raise AgentError("No cohort selected – run the Cohort Analysis Agent first or pass a cohort.")


def scores_path(model_id: str):
    return get_config().models_dir / f"{model_id}_scores.parquet"


def load_scores(model_id: str | None = None) -> pd.DataFrame:
    if model_id is None:
        m = reg.active_model()
        if m is None:
            raise AgentError("No active model – run the Risk Discovery Agent first.")
        model_id = m["model_id"]
    p = scores_path(model_id)
    if not p.exists():
        raise AgentError(f"Scores for model {model_id} not found")
    return pd.read_parquet(p)


def segment_loans(segment_id: int, model_id: str | None = None, split: str | None = None) -> pd.DataFrame:
    s = load_scores(model_id)
    s = s[s["segment_id"] == segment_id]
    return s[s["split"] == split] if split else s


class RiskDiscoveryAgent(BaseAgent):
    name = "risk_discovery"
    title = "Risk Discovery & ML Agent"
    description = "Chronologically validated decision tree, leaf segments, feature importance and IV/WoE."
    writes_state = True
    retry_safe = True

    def execute(self, ctx: dict, cohort: str | None = None, max_depth: int | None = None,
                min_samples_leaf: int | None = None, force_retrain: bool = False, **_) -> AgentResult:
        cfg = get_config()
        t0 = time.perf_counter()
        steps: list[dict] = []
        cohort = resolve_cohort(ctx, cohort)
        data_hash = ctx.get("data_hash") or current_data_hash(cfg)

        self.step(f"Building dynamic target for cohort {cohort}", steps, t0)
        ads = get_processed_datasets(cfg)
        lf = loan_cohort_frame(ads.loan_master, ads.repayment_history, cohort)
        qt = cohort_quarter_table(lf, cohort, ads.observation_cutoff)
        mature = qt.loc[qt["is_mature"], "quarter"].tolist()
        try:
            md = build_modeling_data(ads.loan_master, lf, mature)
        except ValueError as exc:
            raise AgentError(str(exc)) from exc
        self.step(f"Chronological split: train {md.train_quarters} ({len(md.train)} loans) → "
                  f"hold-out {md.valid_quarters} ({len(md.valid)} loans)", steps, t0)

        tc = cfg.tree
        params = {"max_depth": max_depth or tc.max_depth, "min_samples_leaf": min_samples_leaf or tc.min_samples_leaf,
                  "class_weight": tc.class_weight, "random_state": tc.random_state, "train_challenger": tc.train_challenger}
        signature = reg.model_signature(data_hash, cohort, md.train_quarters, md.valid_quarters, params)
        model_id = reg.model_id_for(cohort, signature)
        retrain, reason = reg.needs_retraining(model_id)
        if force_retrain:
            retrain, reason = True, "Retraining forced by user"

        if retrain:
            self.step(f"Training DecisionTreeClassifier ({reason})", steps, t0)
            try:
                tm = train_models(md, **params)
            except ValueError as exc:
                raise AgentError(str(exc)) from exc
            Xtr = tm.spec.transform(md.train)
            leaf_pd = leaf_default_rates(tm.tree, Xtr, md.train["target"].astype(int))
            bundle = {"model_id": model_id, "signature": signature, "cohort": cohort, "tree": tm.tree,
                      "challenger": tm.challenger, "spec": tm.spec.to_dict(), "leaf_pd": leaf_pd, "params": tm.params,
                      "train_quarters": md.train_quarters, "valid_quarters": md.valid_quarters,
                      "data_hash": data_hash, "trained_at": utcnow()}
            artifact = reg.save_bundle(model_id, bundle)
        else:
            self.step(f"Reusing registered model ({reason})", steps, t0)
            bundle = reg.load_bundle(model_id)
            tm = TrainedModels(spec=FeatureSpec.from_dict(bundle["spec"]), tree=bundle["tree"],
                               challenger=bundle["challenger"], params=bundle["params"])
            leaf_pd = bundle["leaf_pd"]
            artifact = reg.find_model(model_id)["artifact_path"]

        self.step("Evaluating on training and held-out quarters", steps, t0)
        spec = tm.spec
        Xtr, Xva = spec.transform(md.train), spec.transform(md.valid)
        ytr, yva = md.train["target"].astype(int), md.valid["target"].astype(int)
        allf = pd.concat([md.train, md.valid, md.unobserved])
        sc = score(tm, allf, leaf_pd)
        allf = allf.join(sc)
        tr_s, va_s = allf.loc[md.train.index], allf.loc[md.valid.index]
        metrics = {
            "train": classification_metrics(ytr, tr_s["predicted_class"].values, tr_s["risk_score"].values),
            "validation": classification_metrics(yva, va_s["predicted_class"].values, va_s["risk_score"].values),
        }
        if tm.challenger is not None:
            metrics["challenger_train"] = classification_metrics(
                ytr, (tr_s["challenger_score"] >= 0.5).astype(int).values, tr_s["challenger_score"].values)
            metrics["challenger_validation"] = classification_metrics(
                yva, (va_s["challenger_score"] >= 0.5).astype(int).values, va_s["challenger_score"].values)
        auc_tr, auc_va = metrics["train"].get("roc_auc"), metrics["validation"].get("roc_auc")
        metrics["overfit_auc_gap"] = None if auc_tr is None or auc_va is None else auc_tr - auc_va

        importance = aggregate_importance(spec, list(Xtr.columns), tm.tree.feature_importances_)
        perm = permutation_importance_by_feature(tm.tree, spec, Xva, yva, tc.random_state)
        ch_imp = aggregate_importance(spec, list(Xtr.columns), tm.challenger.feature_importances_) if tm.challenger is not None else []

        self.step("Extracting decision paths and leaf segments", steps, t0)
        nodes, segments = extract_tree(tm.tree, spec, Xtr, ytr, Xva, yva)
        allf["segment_id"] = assign_segments(allf["leaf_node"], segments)
        overall_rate = float(ytr.mean())
        profiles = segment_profiles(allf, segments, overall_rate)
        n_leaf_train = sum(p["train"]["loans"] for p in profiles)
        if n_leaf_train != len(md.train):
            raise AgentError(f"Segment reconciliation failed: {n_leaf_train} loans in leaves vs {len(md.train)} trained")

        self.step("Computing Information Value / Weight of Evidence (training data)", steps, t0)
        ivs = information_values(md.train, "target", IV_FEATURES)

        self.step("Persisting model, segments and loan scores", steps, t0)
        allf["split"] = allf["split"].astype(str)
        allf[[c for c in SCORE_COLUMNS if c in allf.columns]].to_parquet(scores_path(model_id), index=False)
        reg.register(model_id, cohort, data_hash, tm.params, spec.source_features,
                     {k: v for k, v in metrics.items() if not isinstance(v, dict) or True},
                     md.train_quarters, md.valid_quarters, artifact, profiles, ctx["run_id"])
        if retrain:
            repo.write_audit("MODEL_TRAINED", ctx.get("actor", "system"), "model", model_id,
                             {"cohort": cohort, "params": tm.params, "train_auc": auc_tr, "valid_auc": auc_va,
                              "reason": reason}, ctx["run_id"])

        va_rows = allf.loc[md.valid.index, ["loan_account_no", "target", "probability_of_default", "risk_score",
                                            "predicted_class", "segment_id"]]
        payload = {
            "model_id": model_id, "signature": signature, "cohort": cohort,
            "cohort_definition": cohort_definition(cohort), "retrained": retrain, "retrain_reason": reason,
            "params": tm.params, "features": spec.source_features, "artifact_path": artifact,
            "split": {"train_quarters": md.train_quarters, "validation_quarters": md.valid_quarters,
                      "train_loans": len(md.train), "train_defaults": int(ytr.sum()),
                      "validation_loans": len(md.valid), "validation_defaults": int(yva.sum()),
                      "unobserved_loans": int((allf["split"] == "unobserved").sum()),
                      "partial_quarter_loans": int((allf["split"] == "partial_quarter").sum()),
                      "train_period": [str(md.train["disbursal_date"].min().date()), str(md.train["disbursal_date"].max().date())],
                      "validation_period": [str(md.valid["disbursal_date"].min().date()), str(md.valid["disbursal_date"].max().date())]},
            "metrics": metrics,
            "feature_importance": importance, "permutation_importance_validation": perm,
            "challenger_importance": ch_imp,
            "calibration_validation": calibration_table(yva, va_s["probability_of_default"]),
            "tree": {"nodes": nodes, "n_leaves": len(segments), "depth": int(tm.tree.get_depth())},
            "segments": profiles, "information_value": ivs,
            "overall_train_default_rate": overall_rate,
            "validation_predictions": va_rows.to_dict(orient="records"),
            "reconciliation": {"loans_in_leaves_train": n_leaf_train, "train_loans": len(md.train),
                               "loans_in_leaves_valid": sum(p["validation"]["loans"] for p in profiles),
                               "validation_loans": len(md.valid)},
            "processing_log": steps, "generated_at": utcnow(), "run_id": ctx["run_id"], "data_hash": data_hash,
        }
        path = save_report(REPORT_NAME, payload)
        ctx.update(model_id=model_id, selected_cohort=cohort, risk_segments=profiles,
                   train_quarters=md.train_quarters, valid_quarters=md.valid_quarters)

        return AgentResult(status="SUCCESS", output={
            "model_id": model_id, "cohort": cohort, "retrained": retrain, "retrain_reason": reason,
            "train_quarters": md.train_quarters, "validation_quarters": md.valid_quarters,
            "train_loans": len(md.train), "validation_loans": len(md.valid),
            "train_auc": auc_tr, "validation_auc": auc_va,
            "validation_recall": metrics["validation"]["recall"], "validation_precision": metrics["validation"]["precision"],
            "n_segments": len(segments), "top_features": [r["feature"] for r in importance[:5]],
            "report_file": path, "processing_log": steps,
        }, findings=self._findings(payload))

    @staticmethod
    def _findings(p: dict) -> list[str]:
        m, s = p["metrics"], p["split"]
        f = [
            f"Cohort {p['cohort']}: trained on {s['train_quarters']} ({s['train_loans']} loans, {s['train_defaults']} defaults); "
            f"held-out {s['validation_quarters']} ({s['validation_loans']} loans, {s['validation_defaults']} defaults).",
            ("Model retrained: " if p["retrained"] else "Model reused: ") + p["retrain_reason"] + ".",
        ]
        va, tr = m["validation"], m["train"]
        if va.get("roc_auc") is not None:
            f.append(f"ROC-AUC train {tr['roc_auc']:.3f} vs held-out {va['roc_auc']:.3f} "
                     f"(gap {m['overfit_auc_gap']:+.3f}); held-out recall {va['recall']:.1%}, precision {va['precision']:.1%}.")
        if "challenger_validation" in m and m["challenger_validation"].get("roc_auc") is not None:
            f.append(f"RandomForest challenger held-out ROC-AUC {m['challenger_validation']['roc_auc']:.3f}.")
        f.append("Top drivers (tree importance): " + ", ".join(
            f"{r['feature']} {r['importance_share']:.0%}" for r in p["feature_importance"][:4] if r["importance"] > 0) + ".")
        f.append("Highest IV: " + ", ".join(f"{r['feature']} {r['iv']:.3f} ({r['strength']})" for r in p["information_value"][:3]) + ".")
        segs = sorted([x for x in p["segments"] if x["train"]["default_rate"] is not None],
                      key=lambda x: x["train"]["default_rate"], reverse=True)
        f.append(f"{len(p['segments'])} leaf segments; overall training default rate {p['overall_train_default_rate']:.2%}.")
        for x in segs[:3]:
            vr = x["validation"]["default_rate"]
            f.append(f"Segment {x['segment_id']}: {x['path_text']} → train {x['train']['default_rate']:.1%} "
                     f"({x['train']['defaults']}/{x['train']['loans']}), held-out "
                     + ("n/a" if vr is None else f"{vr:.1%} ({x['validation']['defaults']}/{x['validation']['loans']})") + ".")
        return f
