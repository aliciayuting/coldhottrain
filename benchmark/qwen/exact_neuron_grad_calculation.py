# ===== Annotated export & plots for top-K neurons/heads (grad & ΔW) =====
import math
from collections import defaultdict
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os, warnings
from neuron_grad_analysis import *

TOP_K = 200  # how many units to list in CSV/plots
GLOBAL_STEP     = 200
OUT_DIR         = "/pscratch/sd/l/lsx/yyt_tmp/Qwen_Qwen2.5-0.5B-tatsu-lab_alpaca/exact_neuron"
CSV_OUT = os.path.join(OUT_DIR, f"topK_neurons_step{GLOBAL_STEP:06d}.csv")
SCATTER_PNG = os.path.join(OUT_DIR, f"topK_scatter_step{GLOBAL_STEP:06d}.png")
BARS_PNG = os.path.join(OUT_DIR, f"top50_bars_step{GLOBAL_STEP:06d}.png")

def _concat_and_rank(records, value_key):
    """Return dataframe with ranks and cumulative fractions for chosen value_key ('grad' or 'delta_total')."""
    df = pd.DataFrame.from_records(records)
    if df.empty:
        return df
    vals = df[value_key].to_numpy(dtype=np.float64)
    order = np.argsort(vals)[::-1]
    df = df.iloc[order].reset_index(drop=True)
    total = float(vals[order].sum())
    df["rank"] = np.arange(1, len(df)+1)
    df["cum"] = np.cumsum(df[value_key].to_numpy(dtype=np.float64))
    df["cum_frac"] = df["cum"] / (total if total > 0 else 1.0)
    df["value_frac"] = df[value_key] / (total if total > 0 else 1.0)
    return df

def collect_annotated_neurons(step: int):
    """
    Recomputes neuron/head energies with annotations:
      kind: 'MLP' or 'ATTN'
      layer: int
      unit: neuron_id (MLP) or head_id (ATTN)
      component: 'mlp_up','mlp_down','q','k','v','o','total'
      grad: value (component or total)
      delta_*: matching ΔW components and totals
    """
    # ---- Grad with annotations ----
    df_idx = pd.read_csv(os.path.join(GRAD_BASE_DIR, "index.csv"))
    rows = df_idx[df_idx["global_step"] == step]
    if rows.empty:
        raise ValueError(f"No entries for global_step={step} in index.csv")

    # First pass: infer (n_heads, n_kv, d_head) per layer (GQA-aware)
    q_rows_layer, kv_rows_layer = {}, {}
    for _, r in rows.iterrows():
        if r["submodule"] != "self_attn": continue
        t = _load_any_tensor(os.path.join(GRAD_BASE_DIR, r["file"]))
        if r["param"] == "q_proj.weight":
            q_rows_layer[int(r["layer"])] = int(t.shape[0])
        elif r["param"] in ("k_proj.weight","v_proj.weight"):
            kv_rows_layer[int(r["layer"])] = max(kv_rows_layer.get(int(r["layer"]),0), int(t.shape[0]))
    head_meta = {}
    for lid, qrows in q_rows_layer.items():
        kvrows = kv_rows_layer.get(lid, None)
        head_meta[lid] = _infer_gqa_meta(qrows, kvrows)

    # Annotated container
    recs = []

    # Second pass: accumulate grad per component
    # For MLP we need to sum row-wise (up) and col-wise (down) into matching neuron ids
    mlp_grad_inc = defaultdict(lambda: None)  # lid -> np.array[d_hidden]
    mlp_grad_out = defaultdict(lambda: None)
    attn_grad_q = defaultdict(lambda: None)   # lid -> np.array[n_heads]
    attn_grad_k = defaultdict(lambda: None)   # after broadcast to n_heads
    attn_grad_v = defaultdict(lambda: None)
    attn_grad_o = defaultdict(lambda: None)

    for _, r in rows.iterrows():
        lid = int(r["layer"]); sub = r["submodule"]; param = r["param"]
        t = _load_any_tensor(os.path.join(GRAD_BASE_DIR, r["file"])).to(torch.float32)

        if sub == "mlp" and param == "up_proj.weight" and t.ndim == 2:
            e = (t.pow(2).sum(dim=1)).cpu().numpy()  # [d_hidden]
            mlp_grad_inc[lid] = e if mlp_grad_inc[lid] is None else (mlp_grad_inc[lid] + e)
        elif sub == "mlp" and param == "down_proj.weight" and t.ndim == 2:
            e = (t.pow(2).sum(dim=0)).cpu().numpy()  # [d_hidden]
            mlp_grad_out[lid] = e if mlp_grad_out[lid] is None else (mlp_grad_out[lid] + e)
        elif sub == "self_attn" and t.ndim == 2 and lid in head_meta:
            n_heads, n_kv, d_head = head_meta[lid]
            if param == "q_proj.weight":
                v = np.zeros(n_heads, dtype=np.float64)
                for h in range(n_heads):
                    v[h] = float((t[h*d_head:(h+1)*d_head, :].pow(2)).sum().item())
                attn_grad_q[lid] = v if attn_grad_q[lid] is None else (attn_grad_q[lid] + v)
            elif param in ("k_proj.weight","v_proj.weight"):
                nk = n_kv
                if t.shape[0] != nk*d_head: continue
                tmp = np.zeros(nk, dtype=np.float64)
                for h in range(nk):
                    tmp[h] = float((t[h*d_head:(h+1)*d_head, :].pow(2)).sum().item())
                group = max(1, n_heads // nk)
                tmp = np.repeat(tmp, group)[:n_heads]
                if param.startswith("k_"):
                    attn_grad_k[lid] = tmp if attn_grad_k[lid] is None else (attn_grad_k[lid] + tmp)
                else:
                    attn_grad_v[lid] = tmp if attn_grad_v[lid] is None else (attn_grad_v[lid] + tmp)
            elif param in ("o_proj.weight","out_proj.weight"):
                v = np.zeros(n_heads, dtype=np.float64)
                for h in range(n_heads):
                    v[h] = float((t[:, h*d_head:(h+1)*d_head].pow(2)).sum().item())
                attn_grad_o[lid] = v if attn_grad_o[lid] is None else (attn_grad_o[lid] + v)

    # Emit annotated grad records
    for lid in sorted(set(list(mlp_grad_inc.keys()) + list(mlp_grad_out.keys()))):
        inc = mlp_grad_inc[lid]; out = mlp_grad_out[lid]
        if inc is None and out is None: continue
        if inc is None: inc = np.zeros_like(out)
        if out is None: out = np.zeros_like(inc)
        total = inc + out
        for i in range(total.size):
            recs.append(dict(kind="MLP", layer=lid, unit=i,
                             component="mlp_up",  grad=float(inc[i]), delta_q=0, delta_k=0, delta_v=0, delta_o=0))
            recs.append(dict(kind="MLP", layer=lid, unit=i,
                             component="mlp_down",grad=float(out[i]), delta_q=0, delta_k=0, delta_v=0, delta_o=0))
            recs.append(dict(kind="MLP", layer=lid, unit=i,
                             component="total",   grad=float(total[i]), delta_q=0, delta_k=0, delta_v=0, delta_o=0))

    for lid, (n_heads, n_kv, d_head) in sorted(head_meta.items()):
        q = attn_grad_q.get(lid); k = attn_grad_k.get(lid); v = attn_grad_v.get(lid); o = attn_grad_o.get(lid)
        if q is None and k is None and v is None and o is None: continue
        q = q if q is not None else np.zeros(n_heads)
        k = k if k is not None else np.zeros(n_heads)
        v = v if v is not None else np.zeros(n_heads)
        o = o if o is not None else np.zeros(n_heads)
        tot = q + k + v + o
        for h in range(n_heads):
            recs.append(dict(kind="ATTN", layer=lid, unit=h, component="q",     grad=float(q[h]),
                             delta_q=0, delta_k=0, delta_v=0, delta_o=0))
            recs.append(dict(kind="ATTN", layer=lid, unit=h, component="k",     grad=float(k[h]),
                             delta_q=0, delta_k=0, delta_v=0, delta_o=0))
            recs.append(dict(kind="ATTN", layer=lid, unit=h, component="v",     grad=float(v[h]),
                             delta_q=0, delta_k=0, delta_v=0, delta_o=0))
            recs.append(dict(kind="ATTN", layer=lid, unit=h, component="o",     grad=float(o[h]),
                             delta_q=0, delta_k=0, delta_v=0, delta_o=0))
            recs.append(dict(kind="ATTN", layer=lid, unit=h, component="total", grad=float(tot[h]),
                             delta_q=0, delta_k=0, delta_v=0, delta_o=0))

    # ---- ΔW with annotations (components) ----
    pre_dir  = os.path.join(WEIGHT_ROOT, f"step{step:06d}_pre")
    post_dir = os.path.join(WEIGHT_ROOT, f"step{step:06d}_post")
    sd_pre  = _load_state_dict_any(pre_dir)
    sd_post = _load_state_dict_any(post_dir)

    # group by layer
    import re
    layer_regex = re.compile(r"\.layers\.(\d+)\.")
    by_layer = defaultdict(dict)
    for k in sd_pre.keys():
        m = layer_regex.search(k)
        if not m: continue
        by_layer[int(m.group(1))][k] = True

    for lid, _ in sorted(by_layer.items()):
        # MLP ΔW
        up = f"model.model.layers.{lid}.mlp.up_proj.weight"
        down = f"model.model.layers.{lid}.mlp.down_proj.weight"
        if up in sd_pre and up in sd_post and down in sd_pre and down in sd_post:
            Dup   = (sd_post[up]   - sd_pre[up]).to(torch.float32)   # [d_hidden, d_model]
            Ddown = (sd_post[down] - sd_pre[down]).to(torch.float32) # [d_model, d_hidden]
            inc = Dup.pow(2).sum(dim=1).cpu().numpy()
            out = Ddown.pow(2).sum(dim=0).cpu().numpy()
            tot = inc + out
            for i in range(tot.size):
                recs.append(dict(kind="MLP", layer=lid, unit=i, component="mlp_up",
                                 grad=0.0, delta_q=float(inc[i]), delta_k=0, delta_v=0, delta_o=0))
                recs.append(dict(kind="MLP", layer=lid, unit=i, component="mlp_down",
                                 grad=0.0, delta_q=float(out[i]), delta_k=0, delta_v=0, delta_o=0))
                recs.append(dict(kind="MLP", layer=lid, unit=i, component="total",
                                 grad=0.0, delta_q=float(tot[i]), delta_k=0, delta_v=0, delta_o=0))

        # ATTN ΔW (GQA-aware)
        qk = f"model.model.layers.{lid}.self_attn.q_proj.weight"
        ok = f"model.model.layers.{lid}.self_attn.o_proj.weight"
        kk = f"model.model.layers.{lid}.self_attn.k_proj.weight"
        vk = f"model.model.layers.{lid}.self_attn.v_proj.weight"
        if qk in sd_pre and qk in sd_post and ok in sd_pre and ok in sd_post:
            Wq_pre, Wq_post = sd_pre[qk], sd_post[qk]
            n_heads, n_kv, d_head = _infer_gqa_meta(int(Wq_pre.shape[0]),
                                                    int(sd_pre[kk].shape[0]) if kk in sd_pre else (int(sd_pre[vk].shape[0]) if vk in sd_pre else None))

            # q rows
            per_q = np.zeros(n_heads, dtype=np.float64)
            Wd = (Wq_post - Wq_pre).to(torch.float32)
            for h in range(n_heads):
                per_q[h] = float((Wd[h*d_head:(h+1)*d_head, :].pow(2)).sum().item())

            # k rows (broadcast)
            per_k = np.zeros(n_heads, dtype=np.float64)
            if kk in sd_pre and kk in sd_post:
                Wd = (sd_post[kk] - sd_pre[kk]).to(torch.float32)
                nk = n_kv
                tmp = np.zeros(nk, dtype=np.float64)
                for h in range(nk):
                    tmp[h] = float((Wd[h*d_head:(h+1)*d_head, :].pow(2)).sum().item())
                per_k = np.repeat(tmp, max(1, n_heads//nk))[:n_heads]

            # v rows (broadcast)
            per_v = np.zeros(n_heads, dtype=np.float64)
            if vk in sd_pre and vk in sd_post:
                Wd = (sd_post[vk] - sd_pre[vk]).to(torch.float32)
                nk = n_kv
                tmp = np.zeros(nk, dtype=np.float64)
                for h in range(nk):
                    tmp[h] = float((Wd[h*d_head:(h+1)*d_head, :].pow(2)).sum().item())
                per_v = np.repeat(tmp, max(1, n_heads//nk))[:n_heads]

            # o cols
            per_o = np.zeros(n_heads, dtype=np.float64)
            Wd = (sd_post[ok] - sd_pre[ok]).to(torch.float32)
            for h in range(n_heads):
                per_o[h] = float((Wd[:, h*d_head:(h+1)*d_head].pow(2)).sum().item())

            tot = per_q + per_k + per_v + per_o
            for h in range(n_heads):
                recs.append(dict(kind="ATTN", layer=lid, unit=h, component="q",
                                 grad=0.0, delta_q=float(per_q[h]), delta_k=0, delta_v=0, delta_o=0))
                recs.append(dict(kind="ATTN", layer=lid, unit=h, component="k",
                                 grad=0.0, delta_q=float(per_k[h]), delta_k=0, delta_v=0, delta_o=0))
                recs.append(dict(kind="ATTN", layer=lid, unit=h, component="v",
                                 grad=0.0, delta_q=float(per_v[h]), delta_k=0, delta_v=0, delta_o=0))
                recs.append(dict(kind="ATTN", layer=lid, unit=h, component="o",
                                 grad=0.0, delta_q=float(per_o[h]), delta_k=0, delta_v=0, delta_o=0))
                recs.append(dict(kind="ATTN", layer=lid, unit=h, component="total",
                                 grad=0.0, delta_q=float(tot[h]), delta_k=0, delta_v=0, delta_o=0))

    # Final tidy: add delta_total column
    for r in recs:
        # For MLP we packed ΔW totals in delta_q; for ATTN we split q/k/v/o
        if r["kind"] == "MLP":
            r["delta_total"] = r["delta_q"] if r["component"] in ("mlp_up","mlp_down","total") else 0.0
        else:
            if r["component"] == "total":
                # for convenience recompute from components later during ranking
                r["delta_total"] = r["delta_q"]  # will overwrite below after we collect per-head rows
            else:
                r["delta_total"] = r["delta_q"]

    # For attention, ensure 'total' rows truly hold q+k+v+o
    df_recs = pd.DataFrame.from_records(recs)
    if not df_recs.empty:
        # Build a per-(ATTN,layer,unit) sum for components
        mask_attn = (df_recs["kind"] == "ATTN")
        comp_sum = (df_recs[mask_attn & df_recs["component"].isin(["q","k","v","o"])]
                    .groupby(["kind","layer","unit"])["delta_total"].sum().rename("delta_attn_sum"))
        df_recs = df_recs.merge(comp_sum, how="left", left_on=["kind","layer","unit"], right_index=True)
        df_recs.loc[mask_attn & (df_recs["component"]=="total"), "delta_total"] = df_recs.loc[mask_attn & (df_recs["component"]=="total"), "delta_attn_sum"].fillna(0.0)
        df_recs.drop(columns=["delta_attn_sum"], inplace=True)

    return df_recs

def export_and_plot_topK(step: int, k: int = TOP_K):
    df = collect_annotated_neurons(step)

    # Two views: totals by neuron/head, for grad and for ΔW
    # 1) totals per structural unit (component == 'total')
    totals = df[df["component"] == "total"].copy()

    # Rank grad totals
    totals_grad = totals.copy()
    totals_grad["value"] = totals_grad["grad"]
    totals_grad = _concat_and_rank(totals_grad, "value")

    # Rank ΔW totals
    totals_delta = totals.copy()
    totals_delta["value"] = totals_delta["delta_total"]
    totals_delta = _concat_and_rank(totals_delta, "value")

    # 2) Export top-K combined table with identity strings
    def _ident(row):
        if row["kind"] == "MLP":
            return f"L{int(row['layer'])}/MLP:{int(row['unit'])}"
        else:
            return f"L{int(row['layer'])}/ATTN:head{int(row['unit'])}"

    out_rows = []

    for name, ranked in [("grad", totals_grad), ("delta", totals_delta)]:
        top = ranked.head(k).copy()
        top["who"] = top.apply(_ident, axis=1)
        top["energy_type"] = name
        out_rows.append(top[["energy_type","kind","layer","unit","who","value","rank","cum_frac","value_frac"]])

    out = pd.concat(out_rows, axis=0).reset_index(drop=True)
    out.to_csv(CSV_OUT, index=False)
    print(f"[CSV] wrote top-{k} neurons/heads with ranks → {CSV_OUT}")

    # ----- Plot 1: rank–cumulative scatter (color by kind) -----
    plt.figure(figsize=(6,4), dpi=200)
    for name, ranked, marker in [
        ("grad", totals_grad, "."),
        ("ΔW",  totals_delta, "x"),
    ]:
        color_map = np.where(ranked["kind"].values == "MLP", 0, 1)  # 0=MLP,1=ATTN
        # Normalize colors for a 2-color scatter (matplotlib auto palette is fine)
        plt.scatter(ranked["rank"].values[:k], ranked["cum_frac"].values[:k],
                    s=14, marker=marker, label=f"{name} (top{k})", alpha=0.8)
    plt.xscale("log")
    plt.xlabel("Rank (log)"); plt.ylabel("Cumulative fraction"); plt.ylim(0,1)
    plt.legend(loc="lower right"); plt.tight_layout()
    plt.savefig(SCATTER_PNG, bbox_inches="tight"); plt.close()
    print(f"[PLOT] rank–cumulative scatter → {SCATTER_PNG}")

    # ----- Plot 2: bar chart of top-50 ΔW with labels -----
    top50 = totals_delta.head(min(50, len(totals_delta))).copy()
    if not top50.empty:
        top50["who"] = top50.apply(_ident, axis=1)
        plt.figure(figsize=(10,4), dpi=200)
        plt.bar(np.arange(len(top50)), top50["value"].values)
        plt.xticks(np.arange(len(top50)), top50["who"].values, rotation=90, fontsize=7)
        plt.ylabel("ΔW energy (L2²)"); plt.tight_layout()
        plt.savefig(BARS_PNG, bbox_inches="tight"); plt.close()
        print(f"[PLOT] top-50 ΔW bars → {BARS_PNG}")
    else:
        print("[PLOT] ΔW totals empty; skipped bar chart.")

# Run the export after your main() if you want:
# export_and_plot_topK(GLOBAL_STEP, TOP_K)