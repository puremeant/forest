"""
Author-style AP Tree stability test in Python.

This ports the core tree-portfolio construction logic from the authors' R code:
- depth = 4 conditional sorting tree
- q_num = 2 median-like splits within each parent node and month
- enumerate all feature-order sequences, e.g. 3^4 for three characteristics
- include all intermediate and final nodes
- remove depth-four single-characteristic nodes
- optional AP-pruning-like LARS/LASSO selection using robust SDF regression

This is designed for prepared yearly chunks like:
Data/data_chunk_files_quantile/LME_OP_Investment/y1964.csv
with columns: yy, mm, ret, size, and characteristic quantiles.

It is not a full drop-in replacement for every plot/table in the paper, but it is
close enough for path/set stability analysis based on the authors' tree logic.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Iterable, Sequence
import json
import math
import numpy as np
import pandas as pd

try:
    from sklearn.linear_model import lars_path
except Exception:  # pragma: no cover
    lars_path = None


@dataclass(frozen=True)
class NodeMeta:
    column: str
    order_digits: str
    node_code: str
    node_depth: int
    effective_order_digits: str
    effective_order_names: tuple[str, ...]
    branch_digits: str

    @property
    def feature_order_label(self) -> str:
        return ">".join(self.effective_order_names)

    @property
    def feature_set_label(self) -> str:
        return "+".join(sorted(set(self.effective_order_names)))

    @property
    def path_label(self) -> str:
        parts = []
        for f, b in zip(self.effective_order_names, self.branch_digits):
            parts.append(f"{f}:{b}")
        return ">".join(parts)


def ntile(values: pd.Series, q: int = 2) -> pd.Series:
    """Mimic dplyr::ntile: split sorted observations into q nearly equal groups.

    Ties are broken by original order, similar to rank(method='first').
    Returns integer labels 1..q aligned with values.index.
    """
    n = len(values)
    if n == 0:
        return pd.Series([], index=values.index, dtype=int)
    order = values.sort_values(kind="mergesort").index
    labels_sorted = (np.floor(np.arange(n) * q / n).astype(int) + 1).clip(1, q)
    out = pd.Series(index=values.index, dtype=int)
    out.loc[order] = labels_sorted
    return out


def load_yearly_chunks(data_dir: str | Path, y_min: int, y_max: int) -> pd.DataFrame:
    data_dir = Path(data_dir)
    frames = []
    for y in range(y_min, y_max + 1):
        p = data_dir / f"y{y}.csv"
        if not p.exists():
            raise FileNotFoundError(f"Missing yearly chunk: {p}")
        frames.append(pd.read_csv(p))
    df = pd.concat(frames, ignore_index=True)
    df = df.sort_values(["yy", "mm", "permno"] if "permno" in df.columns else ["yy", "mm"]).reset_index(drop=True)
    return df


def monthly_tree_codes(df_month: pd.DataFrame, feature_order: Sequence[str], q_num: int = 2) -> dict[int, pd.Series]:
    """Return node code for each level for one month.

    Level 0 code is always '1'. Level 1 codes are '11' or '12'.
    Level 2 codes are '111', '112', '121', '122', etc.
    """
    codes_by_level: dict[int, pd.Series] = {0: pd.Series("1", index=df_month.index, dtype="object")}
    current_groups = {"1": df_month.index}

    for level, feat in enumerate(feature_order, start=1):
        next_codes = pd.Series(index=df_month.index, dtype="object")
        next_groups = {}
        for parent_code, idx in current_groups.items():
            if len(idx) == 0:
                continue
            labels = ntile(df_month.loc[idx, feat], q_num)
            for val in range(1, q_num + 1):
                child_idx = labels.index[labels == val]
                child_code = parent_code + str(val)
                next_codes.loc[child_idx] = child_code
                next_groups[child_code] = child_idx
        codes_by_level[level] = next_codes
        current_groups = next_groups
    return codes_by_level


def value_weighted_return(ret: pd.Series, size: pd.Series) -> float:
    denom = size.sum()
    if denom == 0 or np.isnan(denom):
        return np.nan
    return float((ret * size).sum() / denom)


def make_author_style_tree_portfolios(
    df: pd.DataFrame,
    features: Sequence[str],
    tree_depth: int = 4,
    q_num: int = 2,
    subtract_rf: Sequence[float] | None = None,
    drop_single_feature_final_nodes: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Construct AP Tree candidate portfolios using authors' deterministic sorting logic.

    Returns
    -------
    ports : DataFrame, T x N monthly portfolio returns/excess returns.
    meta  : DataFrame with one row per portfolio column.
    """
    required = {"yy", "mm", "ret", "size", *features}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    months = df[["yy", "mm"]].drop_duplicates().sort_values(["yy", "mm"]).reset_index(drop=True)
    month_keys = [tuple(x) for x in months[["yy", "mm"]].to_numpy()]

    port_cols: dict[str, list[float]] = {}
    meta_records: list[NodeMeta] = []

    feature_digits = {str(i + 1): feat for i, feat in enumerate(features)}

    for order_digits_tuple in product(range(1, len(features) + 1), repeat=tree_depth):
        order_digits = "".join(map(str, order_digits_tuple))
        feature_order = [features[i - 1] for i in order_digits_tuple]

        # Store returns for each node code across months.
        returns_by_node: dict[str, list[float]] = {}

        for yy, mm in month_keys:
            df_m = df[(df["yy"] == yy) & (df["mm"] == mm)]
            codes_by_level = monthly_tree_codes(df_m, feature_order, q_num=q_num)

            for level, codes in codes_by_level.items():
                for node_code in sorted(codes.dropna().unique()):
                    returns_by_node.setdefault(node_code, [])
                    idx = codes.index[codes == node_code]
                    returns_by_node[node_code].append(value_weighted_return(df_m.loc[idx, "ret"], df_m.loc[idx, "size"]))

            # If a node is empty in a month, fill with nan to keep alignment.
            # With ntile this should rarely happen unless parent groups are tiny.
            for node_code, vals in returns_by_node.items():
                if len(vals) < len([k for k in month_keys if k <= (yy, mm)]):
                    vals.append(np.nan)

        for node_code, vals in returns_by_node.items():
            if len(vals) != len(month_keys):
                # Align defensively; this should not occur with normal data.
                vals = vals + [np.nan] * (len(month_keys) - len(vals))
            node_depth = len(node_code) - 1
            if drop_single_feature_final_nodes and node_depth == tree_depth and len(set(order_digits)) == 1:
                continue
            effective_order_digits = order_digits[:node_depth]
            effective_order_names = tuple(feature_digits[d] for d in effective_order_digits)
            branch_digits = node_code[1:]
            col = f"{order_digits}.{node_code}"
            port_cols[col] = vals
            meta_records.append(
                NodeMeta(
                    column=col,
                    order_digits=order_digits,
                    node_code=node_code,
                    node_depth=node_depth,
                    effective_order_digits=effective_order_digits,
                    effective_order_names=effective_order_names,
                    branch_digits=branch_digits,
                )
            )

    ports = pd.DataFrame(port_cols)
    ports.index = pd.MultiIndex.from_frame(months, names=["yy", "mm"])

    # Deduplicate exactly equal columns, as in the authors' R code.
    transposed = ports.T
    keep_mask = ~transposed.duplicated()
    ports = ports.loc[:, keep_mask.to_numpy()]
    kept_cols = set(ports.columns)
    meta = pd.DataFrame([m.__dict__ | {
        "feature_order_label": m.feature_order_label,
        "feature_set_label": m.feature_set_label,
        "path_label": m.path_label,
    } for m in meta_records])
    meta = meta[meta["column"].isin(kept_cols)].reset_index(drop=True)

    if subtract_rf is not None:
        rf = np.asarray(list(subtract_rf), dtype=float)
        if len(rf) != len(ports):
            raise ValueError(f"rf length {len(rf)} != number of months {len(ports)}")
        ports = ports.sub(rf / 100.0, axis=0)

    return ports, meta


def robust_lars_select(
    ports: pd.DataFrame,
    meta: pd.DataFrame,
    train_idx: np.ndarray,
    valid_idx: np.ndarray | None = None,
    port_n: int = 10,
    lambda0: float = 0.15,
    lambda2: float = 1e-8,
    kmin: int = 1,
    kmax: int = 50,
) -> tuple[pd.Index, pd.Series, dict]:
    """Approximate the authors' AP-pruning LARS step in Python.

    This follows lasso_valid_par_full.R closely enough for stability analysis.
    It returns selected portfolio columns and normalized weights.
    """
    if lars_path is None:
        raise ImportError("scikit-learn is required for lars_path")

    X_train_ports = ports.iloc[train_idx].to_numpy(dtype=float)
    colnames = ports.columns
    node_depth = meta.set_index("column").loc[colnames, "node_depth"].to_numpy(dtype=float)
    adj_w = 1.0 / np.sqrt(2.0 ** node_depth)

    # Authors multiply each portfolio by adj_w before estimating moments.
    X_train_adj = X_train_ports * adj_w
    mu = np.nanmean(X_train_adj, axis=0)
    sigma = np.cov(X_train_adj, rowvar=False)
    mu_bar = float(np.mean(mu))

    # Eigen-decomposition, keep positive eigenvalues.
    vals, vecs = np.linalg.eigh(sigma)
    order = np.argsort(vals)[::-1]
    vals = vals[order]
    vecs = vecs[:, order]
    keep = vals > 1e-10
    vals = vals[keep]
    vecs = vecs[:, keep]

    sigma_tilde = vecs @ np.diag(np.sqrt(vals)) @ vecs.T
    y = vecs @ np.diag(1.0 / np.sqrt(vals)) @ vecs.T @ (mu + lambda0 * mu_bar)

    p = ports.shape[1]
    X_aug = np.vstack([sigma_tilde, np.sqrt(lambda2) * np.eye(p)])
    y_aug = np.concatenate([y, np.zeros(p)])

    _, _, coef_path = lars_path(X_aug, y_aug, method="lasso", verbose=False)
    # coef_path shape: p x n_steps
    records = []
    for step in range(coef_path.shape[1]):
        coef = coef_path[:, step].copy()
        k = int(np.sum(np.abs(coef) > 1e-12))
        if k < kmin or k > kmax:
            continue
        # Convert like authors: b = coef * adj_w; normalize.
        b = coef * adj_w
        denom = abs(np.sum(b))
        if denom > 0:
            b = b / denom
        if valid_idx is not None and len(valid_idx) > 0:
            sdf = ports.iloc[valid_idx].to_numpy(dtype=float) @ (b / adj_w)
            sr = float(np.nanmean(sdf) / np.nanstd(sdf, ddof=1)) if np.nanstd(sdf, ddof=1) > 0 else -np.inf
        else:
            sdf = X_train_ports @ (b / adj_w)
            sr = float(np.nanmean(sdf) / np.nanstd(sdf, ddof=1)) if np.nanstd(sdf, ddof=1) > 0 else -np.inf
        records.append((abs(k - port_n), -sr, step, k, b))

    if not records:
        raise RuntimeError("No LARS path step satisfied kmin/kmax")

    # Prefer exactly port_n if available; within same distance, maximize validation SR.
    records.sort(key=lambda x: (x[0], x[1]))
    _, neg_sr, step, k, b = records[0]
    selected_mask = np.abs(b) > 1e-12
    selected_cols = colnames[selected_mask]
    weights = pd.Series(b[selected_mask], index=selected_cols)
    info = {"step": step, "portsN": k, "selection_sr": -neg_sr}
    return selected_cols, weights, info


def jaccard(a: Iterable[str], b: Iterable[str]) -> float:
    A, B = set(a), set(b)
    if not A and not B:
        return 1.0
    return len(A & B) / len(A | B)


def summarize_window_stability(selection_meta: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Pairwise Jaccard similarity across selected paths/orders/sets."""
    rows = []
    names = list(selection_meta)
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a = selection_meta[names[i]]
            b = selection_meta[names[j]]
            rows.append({
                "window_a": names[i],
                "window_b": names[j],
                "jaccard_path": jaccard(a["path_label"], b["path_label"]),
                "jaccard_feature_order": jaccard(a["feature_order_label"], b["feature_order_label"]),
                "jaccard_feature_set": jaccard(a["feature_set_label"], b["feature_set_label"]),
            })
    return pd.DataFrame(rows)


if __name__ == "__main__":
    # Example using the authors' provided intermediate data.
    base = Path("Replication Code-20210833/Data")
    data_dir = base / "data_chunk_files_quantile" / "LME_OP_Investment"
    rf_path = base / "factor" / "rf_factor.csv"

    df = load_yearly_chunks(data_dir, 1964, 2016)
    rf = pd.read_csv(rf_path, header=None)[0].to_numpy()
    ports, meta = make_author_style_tree_portfolios(
        df,
        features=["LME", "OP", "Investment"],
        tree_depth=4,
        q_num=2,
        subtract_rf=rf,
    )

    out = Path("author_style_tree_stability_out")
    out.mkdir(exist_ok=True)
    ports.to_csv(out / "author_style_tree_ports.csv")
    meta.to_csv(out / "author_style_tree_meta.csv", index=False)

    # Example rolling windows: previous 360 months train/valid, next 36 months testing.
    # Validation is the last 120 months within the previous 360 months.
    selection_meta = {}
    T = len(ports)
    for test_start in range(360, T - 36 + 1, 12):
        train_valid = np.arange(test_start - 360, test_start)
        train_idx = train_valid[:240]
        valid_idx = train_valid[240:360]
        selected, weights, info = robust_lars_select(
            ports,
            meta,
            train_idx=train_idx,
            valid_idx=valid_idx,
            port_n=10,
            lambda0=0.15,
            lambda2=1e-8,
            kmin=1,
            kmax=50,
        )
        win_name = f"{ports.index[test_start][0]}-{ports.index[test_start][1]:02d}"
        sel_meta = meta[meta["column"].isin(selected)].copy()
        sel_meta["weight"] = sel_meta["column"].map(weights)
        sel_meta["portsN"] = info["portsN"]
        sel_meta["selection_sr"] = info["selection_sr"]
        selection_meta[win_name] = sel_meta
        sel_meta.to_csv(out / f"selected_meta_{win_name}.csv", index=False)

    jac = summarize_window_stability(selection_meta)
    jac.to_csv(out / "window_jaccard_similarity.csv", index=False)
    report = pd.DataFrame({
        "measure": ["path", "feature_order", "feature_set"],
        "avg_jaccard": [jac["jaccard_path"].mean(), jac["jaccard_feature_order"].mean(), jac["jaccard_feature_set"].mean()],
        "min_jaccard": [jac["jaccard_path"].min(), jac["jaccard_feature_order"].min(), jac["jaccard_feature_set"].min()],
    })
    report.to_csv(out / "stability_report.csv", index=False)
    print(report)
