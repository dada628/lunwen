"""实验运行器：按大纲【九】的执行顺序跑完整实验链。

执行阶段（顺序不可颠倒，因为超参必须先在 A 格调好）
--------------------------------------------------
  stage=prepare     生成/加载数据与划分索引，写出数据快照信息
  stage=tune        仅条件 A：各模型调参，结果写 outputs/results/hyperparams_*.json
  stage=factors     条件 A/B/C/D（+可选 S1'）：主表 + 表 4
  stage=size_control B' 规模对照（LightGBM 与 BERT 各一次）
  stage=ablation    provenance 消融（结构化端 + 文本端）与弱基线自曝
  stage=all         依次执行以上全部

用法
----
    python scripts/run_experiments.py --stage all --smoke        # 合成数据快速自测
    python scripts/run_experiments.py --stage tune --models bert # 只调 BERT
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from task2difficulty import config as C          # noqa: E402
from task2difficulty import dataset as D          # noqa: E402
from task2difficulty import experiments as E      # noqa: E402
from task2difficulty import metrics as M          # noqa: E402
from task2difficulty import models_deep as MD     # noqa: E402
from task2difficulty import synthetic as S        # noqa: E402
from task2difficulty import vectorizers as V      # noqa: E402
from task2difficulty import vectorizers as V      # noqa: E402


# ---------------------------------------------------------------- 工具


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def ensure_dirs() -> None:
    for d in (C.DATA_DIR, C.RAW_DIR, C.PROCESSED_DIR,
              C.OUTPUTS_DIR, C.RESULTS_DIR, C.FIGURES_DIR, C.REPORTS_DIR):
        d.mkdir(parents=True, exist_ok=True)


def tag_columns_of(ds: D.DifficultyDataset) -> list[str]:
    return [c for c in ds.group_columns("tag") if c.startswith(C.TAG_MULTIHOT_PREFIX)]


# ---------------------------------------------------------------- 数据


def load_or_make_dataset(smoke: bool, n_smoke: int = 1500) -> D.DifficultyDataset:
    """--smoke 用合成数据，否则读真实数据集（需先跑抓取与清洗）。"""
    if smoke:
        path = C.PROCESSED_DIR / "dataset_synthetic.csv"
        if not path.exists():
            log(f"生成合成数据集 -> {path}")
            df = S.make_synthetic(S.SyntheticSpec(n=n_smoke, seed=0))
            path.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(path, index=False, encoding="utf-8-sig")
        return D.load_dataset(path)

    if not C.DATASET_PATH.exists():
        raise FileNotFoundError(
            f"未找到真实数据集 {C.DATASET_PATH}。请先运行数据管线：\n"
            f"  python scripts/fetch_and_build.py\n"
            f"或用 --smoke 在合成数据上自测。"
        )
    return D.load_dataset(C.DATASET_PATH)


def get_split_index(ds: D.DifficultyDataset, smoke: bool) -> D.SplitIndex:
    path = C.SPLIT_INDEX_PATH if not smoke else (C.PROCESSED_DIR / "split_index_synthetic.csv")
    if path.exists():
        si = D.load_split_index(path)
        log(f"载入已有划分索引 {path.name}（共用同一索引文件）")
    else:
        si = D.make_split_index(ds.meta["year"], ds.meta["problem_key"])
        D.save_split_index(si, path)
        log(f"生成划分索引 -> {path.name}")
    for kind in (C.SPLIT_RANDOM, C.SPLIT_TEMPORAL):
        r = si.ratios(kind)
        log(f"  {kind:9s} 训练 {r['train']:.1%} / 验证 {r['valid']:.1%} / 测试 {r['test']:.1%}")
    for w in D.check_split_ratios(si):
        log(f"  [warn] {w}")
    return si


# ---------------------------------------------------------------- 模型工厂分发


def text_assets(pkg: E.DataPackage, model_name: str, cfg: MD.TrainConfig,
                smoke: bool) -> dict:
    """按模型准备文本侧资产。**词表/tokenizer 均只用训练集拟合**。"""
    out: dict = {}
    if model_name in ("textcnn", "bilstm_attn"):
        vocab = V.SimpleVocab(max_vocab=20000 if smoke else 60000,
                              min_freq=1 if smoke else 2,
                              max_length=cfg.seq_length)
        vocab.fit(pkg.ds_train.text)
        out["text_vocab"] = vocab
    if model_name in ("bert", "fusion"):
        tok = V.BertTokenizerWrapper(model_name="bert-base-uncased",
                                     max_length=cfg.max_length).fit(pkg.ds_train.text)
        out["bert_tokenizer"] = tok
    return out


def run_one(model_name: str, pkg: E.DataPackage, params: dict, seed: int,
            smoke: bool, device=None, log_fn=None,
            allow_test_repeat: bool = False) -> E.RunResult:
    """统一入口：按模型分发到传统侧或深度侧。"""
    if model_name in C.TRADITIONAL_MODELS:
        return E.run_tabular(model_name, pkg, params=params, seed=seed,
                             tag_columns=tag_columns_of(pkg.ds_train),
                             allow_test_repeat=allow_test_repeat)
    cfg = train_config_from(params, smoke=smoke)
    assets = text_assets(pkg, model_name, cfg, smoke)
    return E.run_deep(model_name, pkg, cfg, seed=seed, device=device,
                      log=log_fn, allow_test_repeat=allow_test_repeat, **assets)


def train_config_from(params: dict, smoke: bool) -> MD.TrainConfig:
    """从超参字典构造 TrainConfig，未给出的项用安全默认值。"""
    allowed = set(MD.TrainConfig.__dataclass_fields__.keys())
    kw = {k: v for k, v in params.items() if k in allowed}
    cfg = MD.TrainConfig(**kw)
    if smoke:
        # 合成数据自测：把成本压到几十秒内
        cfg.epochs = min(cfg.epochs, 2)
        cfg.patience = 1
        cfg.batch_size = min(cfg.batch_size, 16)
        cfg.max_length = min(cfg.max_length, 128)
        cfg.seq_length = min(cfg.seq_length, 128)
    return cfg


# ---------------------------------------------------------------- 阶段：prepare


def stage_prepare(ds: D.DifficultyDataset, si: D.SplitIndex, smoke: bool) -> pd.DataFrame:
    log("阶段 prepare：数据快照与表 1 素材")
    summary = D.dataset_summary(ds)
    E.assert_no_stat_leak(ds, C.STAT_FULL)
    rows = []
    for kind in (C.SPLIT_RANDOM, C.SPLIT_TEMPORAL):
        r = si.ratios(kind)
        rows.append({"split": kind, **{f"{k}_ratio": v for k, v in r.items()},
                     **{k: int(round(v * len(ds))) for k, v in r.items()}})
    tbl = pd.DataFrame(rows)
    out = C.RESULTS_DIR / ("table1_dataset_synthetic.csv" if smoke else "table1_dataset.csv")
    tbl.to_csv(out, index=False, encoding="utf-8-sig")
    (C.RESULTS_DIR / ("dataset_summary_synthetic.json" if smoke else "dataset_summary.json")
     ).write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=str),
                  encoding="utf-8")
    log(f"  题目数 {summary['n_problems']}，rating {summary['rating_min']:.0f}~"
        f"{summary['rating_max']:.0f}，快照 {summary['snapshot_date']}")
    log(f"  两套划分三段规模 -> {out.name}")
    return tbl


# ---------------------------------------------------------------- 阶段：tune


def stage_tune(ds, si, models, smoke: bool, seeds_for_tune=(0,), device=None,
               force: bool = False) -> None:
    log("阶段 tune：**只在条件 A 上**调参（调好后 B/C/D 格复用，不再调）")
    cond_a = C.CONDITIONS["A"]
    for model_name in models:
        p = E.hyperparams_path(model_name)
        if p.exists() and not force:
            log(f"  {model_name:14s} 已有 {p.name}，跳过（--force 可重调）")
            continue
        space = E.search_space_for(model_name)
        budget = E.budget_for(model_name)
        cands = E.sample_params(space, budget, seed=0)
        if not cands:
            log(f"  {model_name:14s} 无搜索空间，记录默认参数")
            E.save_hyperparams(model_name, {}, tuned_on="A")
            continue

        need_tfidf = model_name in ("ridge", "svr", "random_forest", "lightgbm",
                                    "ordinal_logit")
        best = None
        t0 = time.time()
        for i, cand in enumerate(cands):
            pkg = E.build_data_package(ds, si, cond_a,
                                       need_tfidf=need_tfidf,
                                       tfidf_max_features=5000 if smoke else 20000)
            try:
                res = run_one(model_name, pkg, cand, seed=seeds_for_tune[0],
                              smoke=smoke, device=device,
                              allow_test_repeat=True)   # 调参阶段测试集不算最终结果
            except Exception as ex:
                log(f"    [{i+1}/{len(cands)}] {cand} 失败: {type(ex).__name__}: {ex}")
                continue
            vm = res.valid_metrics["mae"]
            if best is None or vm < best[0]:
                best = (vm, cand, res)
                log(f"    [{i+1}/{len(cands)}] 新最优 valid MAE={vm:.2f}  {cand}")
            else:
                log(f"    [{i+1}/{len(cands)}] valid MAE={vm:.2f}")

        if best is None:
            log(f"  {model_name:14s} 全部候选失败，跳过")
            continue
        E.save_hyperparams(
            model_name, best[1], tuned_on="A",
            extra={"best_valid_mae": best[0], "n_candidates": len(cands),
                   "tune_seconds": round(time.time() - t0, 1),
                   "budget_family": C.MODEL_BUDGET_FAMILY[model_name],
                   "budget": budget},
        )
        log(f"  {model_name:14s} 最优 valid MAE={best[0]:.2f}  用时 {time.time()-t0:.0f}s -> "
            f"{p.name}")


# ---------------------------------------------------------------- 阶段：factors


def stage_factors(ds, si, models, smoke: bool, seeds, device=None,
                  conditions=("A", "B", "C", "D"), text_modes=("clean",),
                  drop_provenance: bool = True,
                  drop_groups=(), keep_groups=None,
                  suffix: str = "") -> pd.DataFrame:
    """条件 A/B/C/D（或指定子集）的主结果。表 3 与表 4 的数据来源。"""
    E.reset_test_guard()
    rows: list[dict] = []
    out = C.RESULTS_DIR / (f"runs{'_synthetic' if smoke else ''}{suffix}.csv")
    for cond_key in conditions:
        cond = C.CONDITIONS[cond_key]
        log(f"  条件 {cond.key}：{cond.positioning}"
            + ("  ⚠ 泄漏格，不可作部署依据" if cond.leaky else ""))
        for model_name in models:
            try:
                params = E.load_hyperparams(model_name)
            except FileNotFoundError:
                params = {}
                log(f"    [warn] {model_name} 无超参文件，用默认参数（应先跑 tune）")
            need_tfidf = model_name in ("ridge", "svr", "random_forest", "lightgbm",
                                        "ordinal_logit")
            for tm in text_modes:
                for seed in seeds:
                    pkg = E.build_data_package(
                        ds, si, cond, need_tfidf=need_tfidf,
                        tfidf_max_features=5000 if smoke else 20000,
                        drop_provenance=drop_provenance,
                        drop_groups=drop_groups, keep_groups=keep_groups,
                        text_mode=tm,
                    )
                    try:
                        res = run_one(model_name, pkg, params, seed=seed,
                                      smoke=smoke, device=device,
                                      allow_test_repeat=True)
                    except Exception as ex:
                        log(f"    {model_name:14s} 失败: {type(ex).__name__}: {ex}")
                        continue
                    row = res.meta_row()
                    row.update({
                        "is_leaky": cond.leaky,
                        "in_main_conclusion": cond.in_main_conclusion,
                        "positioning": cond.positioning,
                        "n_struct_features": res.extra.get("n_struct_features"),
                        "text_mode": tm,
                        "stage": "factors" + suffix,
                    })
                    rows.append(row)
                    # ★ 每行都落盘一次：factors 阶段要跑数小时（BERT 占大头），
                    # 若只在阶段末尾写文件，中途任何一次崩溃（显存不足、进程被杀）
                    # 都会丢掉整段工作。CSV 只有百来行，反复写几乎不花时间。
                    pd.DataFrame(rows).to_csv(out, index=False, encoding="utf-8-sig")
                    # 逐样本预测另存 npz：配对 bootstrap 与误差分析需要它。
                    # 必须同时写入测试集题目键，否则消费方只能靠行序重建索引，
                    # 而数据集/索引文件一变就会静默错位。
                    E.save_predictions(
                        res, suffix=f"_{suffix}" if suffix else "",
                        problem_key=pkg.ds_test.meta["problem_key"].tolist(),
                        split=cond.split,
                    )
                    log(f"    {model_name:14s} {tm:8s} seed={seed:<3d} "
                        f"test MAE={res.test_metrics['mae']:7.2f} "
                        f"±200={res.test_metrics['hit200']:.3f} "
                        f"macroF1={res.test_metrics['macro_f1']:.3f} "
                        f"QWK={res.test_metrics['qwk']:.3f} "
                        f"ρ={res.test_metrics['spearman']:.3f} "
                        f"({res.train_seconds:.0f}s)")
    df = pd.DataFrame(rows)
    df.to_csv(out, index=False, encoding="utf-8-sig")
    log(f"  原始结果 -> {out.name}（{len(df)} 行）")
    return df


# ---------------------------------------------------------------- 阶段：size_control


def stage_size_control(ds, si, smoke: bool, device=None, seeds=(0,)) -> pd.DataFrame:
    """B' 规模对照：随机划分训练集下采样到与时间划分**实际**同规模。

    大纲四·（二）：B−D 混杂训练集规模。B' 只控制**训练规模**，
    仍含训练集构成与测试期分布迁移，故 B'−D 表述为
    "剔除训练量差异后的时间效应（仍含训练集构成与测试期分布迁移）"，**不得**写成"纯净"。

    ⚠ 下采样目标必须取**本次划分索引实测**的时间划分训练占比，
    不能用 `config.EXPECTED_TEMPORAL_RATIOS` 的 0.655——那是**全量 11152 题**的值，
    而抽样数据（如 391 题）的时间划分训练占比实测为 70.6%。
    若照搬 0.655，B' 训练集只有 256 题、而 D 格有 276 题，**并未真正对齐**，
    规模对照就失去了意义（实测踩过）。
    """
    log("阶段 size_control：B′ = 随机划分·训练集下采样至时间划分同规模")
    cond_b = C.CONDITIONS["B"]
    n_total = len(ds)

    # ---- 目标规模 = 时间划分训练段的**实测**规模 ----
    n_time_train = int(si.mask(C.SPLIT_TEMPORAL, "train").sum())
    target_ratio = n_time_train / max(1, n_total)
    n_target = n_time_train
    expected = C.EXPECTED_TEMPORAL_RATIOS["train"]
    if abs(target_ratio - expected) > 0.005:
        log(f"  [note] 时间划分训练占比实测 {target_ratio:.1%}，"
            f"与大纲全量值 {expected:.1%} 相差 {abs(target_ratio-expected)*100:.1f} pp；"
            f"B′ 按**实测**规模对齐（而非全量值）")
    rng = np.random.default_rng(C.SPLIT_SEED + 1)

    idx_all = si.indices(cond_b.split, "train")          # 全量数据中的位置索引
    # 下采样得到 keep_local（训练子集内的位置）与 keep_global（全量数据中的位置）：
    # 二者必须都保留，否则 select() 的索引基准会错位。
    if len(idx_all) > n_target:
        pick_local = rng.choice(len(idx_all), size=n_target, replace=False)
        pick_local.sort()
        keep_local = idx_all[pick_local]
    else:
        keep_local = idx_all.copy()
    keep_global = keep_local.copy()
    log(f"  随机划分训练集 {len(idx_all)} -> {len(keep_local)} 题（占全量 "
        f"{len(keep_local)/n_total:.1%}，与时间划分训练段 {n_time_train} 题对齐）")

    models = [m for m in ("lightgbm", "bert") if not smoke or m != "bert"]
    rows: list[dict] = []
    # 按条件处理后的全量数据集，作为三个子集统一的索引基准
    ds_cond = ds.with_feature_condition(cond_b.stat, drop_provenance=True)

    for model_name in models:
        try:
            params = E.load_hyperparams(model_name)
        except FileNotFoundError:
            params = {}
        pkg = E.build_data_package(ds, si, cond_b,
                                   need_tfidf=(model_name == "lightgbm"),
                                   tfidf_max_features=5000 if smoke else 20000)
        # 下采样只作用于训练集；验证/测试集与 B 格完全一致，保证可比
        pkg.ds_train = ds_cond.select(keep_global)
        pkg._struct_cache.clear()
        # TF-IDF 必须重新在**下采样后的训练集**上拟合（否则训练用了全量词表，规模对照不成立）
        if model_name == "lightgbm":
            vec = V.TfidfTextVectorizer(max_features=5000 if smoke else 20000)
            pkg.tfidf_train = vec.fit_transform(pkg.ds_train.text)
            pkg.tfidf_valid = vec.transform(pkg.ds_valid.text)
            pkg.tfidf_test = vec.transform(pkg.ds_test.text)
            pkg.vocab_size = vec.vocabulary_size
        log(f"  {model_name:14s} 训练集 {len(idx_all)} -> {len(pkg.ds_train)} 题")
        for seed in seeds:
            try:
                res = run_one(model_name, pkg, params, seed=seed, smoke=smoke,
                              device=device, allow_test_repeat=True)
            except Exception as ex:
                log(f"    {model_name} 失败: {type(ex).__name__}: {ex}")
                continue
            row = res.meta_row()
            row.update({
                "condition": "B'", "stage": "size_control",
                "train_ratio": len(keep_local) / n_total,
                "size_control_note": C.SIZE_CONTROL_NOTE,
            })
            rows.append(row)
            log(f"    {model_name:14s} B′ test MAE={res.test_metrics['mae']:.2f}")
    df = pd.DataFrame(rows)
    out = C.RESULTS_DIR / (f"runs_bprime{'_synthetic' if smoke else ''}.csv")
    df.to_csv(out, index=False, encoding="utf-8-sig")
    log(f"  -> {out.name}（{len(df)} 行）")
    return df


# ---------------------------------------------------------------- 阶段：data_efficiency


def stage_data_efficiency(ds, si, smoke: bool, device=None,
                          models=("lightgbm", "bert"),
                          fractions=None) -> pd.DataFrame:
    """实验 5（大纲四·（六））：数据效率曲线。

    在条件 D（最真实的新题场景）下按训练集比例 10%/25%/50%/100% 分别重训，
    记录测试集 MAE。预期：小数据树模型占优，随数据增大深度模型逼近——
    用于解释"为何本任务规模下传统模型不落下风"。

    ⚠ 三点必须做对，否则曲线不可解释：
    1. **只下采样训练集**，验证集与测试集全程不变（否则每点的评估口径不同，曲线无意义）；
    2. **TF-IDF 要在每个比例下的训练子集上重新 fit**，否则"训练量"名义变小、
       实际词表仍来自全量，曲线被人为抬高；
    3. 每个比例用**同一批随机抽样种子**，使各点之间的差异只来自训练量。
    """
    log("阶段 data_efficiency：条件 D 下按训练比例重训（大纲四·（六））")
    fracs = list(fractions or C.DATA_EFFICIENCY_FRACTIONS)
    if smoke:
        models = [m for m in models if m != "bert"]
    cond = C.CONDITIONS["D"]
    n_total = len(ds)
    ds_cond = ds.with_feature_condition(cond.stat, drop_provenance=True)
    idx_train_all = si.indices(cond.split, "train")
    rows: list[dict] = []

    for frac in fracs:
        n_use = max(50, int(round(len(idx_train_all) * frac)))
        n_use = min(n_use, len(idx_train_all))
        rng = np.random.default_rng(C.SPLIT_SEED + 7)
        pick = np.sort(rng.choice(len(idx_train_all), size=n_use, replace=False))
        keep_global = idx_train_all[pick]
        log(f"  训练比例 {frac:.0%}：{len(idx_train_all)} -> {n_use} 题"
            f"（占全量 {n_use/n_total:.1%}）")

        for model_name in models:
            try:
                params = E.load_hyperparams(model_name)
            except FileNotFoundError:
                params = {}
            need_tfidf = model_name in ("ridge", "svr", "random_forest", "lightgbm",
                                        "ordinal_logit")
            pkg = E.build_data_package(ds, si, cond,
                                       need_tfidf=need_tfidf,
                                       tfidf_max_features=5000 if smoke else 20000)
            pkg.ds_train = ds_cond.select(keep_global)
            pkg._struct_cache.clear()
            # ⚠ 文本降维的投影方向也是"训练集相关"的：下采样后必须重拟合，
            # 否则曲线会被"用全量训练集算出的 SVD 方向"人为抬高。
            # 当前默认模型（lightgbm/bert）不走 SVD，这行是为将来把 SVR 纳入曲线留的护栏。
            pkg._text_svd_cache.clear()
            # TF-IDF 必须在**下采样后的训练集**上重新拟合（见 docstring 第 2 点）
            if need_tfidf:
                vec = V.TfidfTextVectorizer(max_features=5000 if smoke else 20000)
                pkg.tfidf_train = vec.fit_transform(pkg.ds_train.text)
                pkg.tfidf_valid = vec.transform(pkg.ds_valid.text)
                pkg.tfidf_test = vec.transform(pkg.ds_test.text)
                pkg.vocab_size = vec.vocabulary_size
            try:
                res = run_one(model_name, pkg, params, seed=0, smoke=smoke,
                              device=device, allow_test_repeat=True)
            except Exception as ex:
                log(f"    {model_name} @ {frac:.0%} 失败: {type(ex).__name__}: {ex}")
                continue
            rows.append({
                "model": model_name,
                "train_fraction": float(frac),
                "n_train": int(n_use),
                "test_mae": float(res.test_metrics["mae"]),
                "test_mae_std": 0.0,          # 单种子；如需误差棒请跑多种子
                "test_hit200": float(res.test_metrics["hit200"]),
                "test_macro_f1": float(res.test_metrics["macro_f1"]),
                "condition": "D",
                "stage": "efficiency",
            })
            log(f"    {model_name:12s} @ {frac:>5.0%}  test MAE={res.test_metrics['mae']:.2f}")

    df = pd.DataFrame(rows)
    out = C.RESULTS_DIR / ("data_efficiency_synthetic.csv" if smoke else "data_efficiency.csv")
    df.to_csv(out, index=False, encoding="utf-8-sig")
    log(f"  -> {out.name}（{len(df)} 行）；图 3 可直接读取该文件")
    return df


# ---------------------------------------------------------------- 阶段：ablation


def stage_ablation(ds, si, smoke: bool, device=None, seeds=(0,)) -> pd.DataFrame:
    """实验 3 + 实验 4 的前置项：provenance 消融、文本端剥离、弱基线自曝。"""
    log("阶段 ablation：provenance 消融（结构+文本两侧）与弱基线自曝")
    rows: list[dict] = []
    out = C.RESULTS_DIR / (f"runs_ablation{'_synthetic' if smoke else ''}.csv")
    cond_d = C.CONDITIONS["D"]     # 最真实的新题场景
    main_models = ["lightgbm", "fusion"]

    # ---- 1) 结构化端：保留来源特征 vs 剔除 ----
    for keep_prov in (False, True):
        tag = "with_provenance" if keep_prov else "no_provenance"
        for model_name in main_models:
            try:
                params = E.load_hyperparams(model_name)
            except FileNotFoundError:
                params = {}
            pkg = E.build_data_package(ds, si, cond_d,
                                       need_tfidf=(model_name == "lightgbm"),
                                       tfidf_max_features=5000 if smoke else 20000,
                                       drop_provenance=not keep_prov)
            for seed in seeds:
                try:
                    res = run_one(model_name, pkg, params, seed=seed, smoke=smoke,
                                  device=device, allow_test_repeat=True)
                except Exception as ex:
                    log(f"    {model_name} {tag} 失败: {type(ex).__name__}: {ex}")
                    continue
                row = res.meta_row()
                row.update({"stage": "ablation_provenance", "variant": tag,
                            "condition": "D"})
                rows.append(row)
                pd.DataFrame(rows).to_csv(out, index=False, encoding="utf-8-sig")
                log(f"    {model_name:10s} {tag:17s} test MAE={res.test_metrics['mae']:.2f} "
                    f"macroF1={res.test_metrics['macro_f1']:.3f}")

    # ---- 2) 文本端：含模板句 vs 去模板句 ----
    for tm in ("clean", "stripped"):
        for model_name in [m for m in ("bert", "fusion") if not smoke]:
            try:
                params = E.load_hyperparams(model_name)
            except FileNotFoundError:
                params = {}
            pkg = E.build_data_package(ds, si, cond_d, text_mode=tm)
            for seed in seeds:
                try:
                    res = run_one(model_name, pkg, params, seed=seed, smoke=smoke,
                                  device=device, allow_test_repeat=True)
                except Exception as ex:
                    log(f"    {model_name} {tm} 失败: {type(ex).__name__}: {ex}")
                    continue
                row = res.meta_row()
                row.update({"stage": "ablation_template", "variant": tm, "condition": "D"})
                rows.append(row)
                pd.DataFrame(rows).to_csv(out, index=False, encoding="utf-8-sig")
                log(f"    {model_name:10s} text={tm:8s} test MAE={res.test_metrics['mae']:.2f}")

    df = pd.DataFrame(rows)
    df.to_csv(out, index=False, encoding="utf-8-sig")
    log(f"  -> {out.name}（{len(df)} 行）")
    return df


# ---------------------------------------------------------------- 主流程


def stage_s1(ds, si, models, smoke: bool, seeds, device=None) -> pd.DataFrame:
    """因子设计的第五格 S1'：时间划分 + **赛时统计**（大纲四·（二）表末行）。

    定位：官方 API 只有"当前累计"统计、无历史接口，因此时间划分下测试集新题
    若用全量快照统计，属于"用最终统计预测最终标签"的乐观泄漏（C 格）。
    S1' 用**比赛结束时的榜单**重建统计——它是历史固定、赛后可立即获得的，
    既非全量累计也非无数据，正好补上"新题 → 存量"的中间档。

    ⚠ 与 A/B/C/D 四格并列呈现，但**不进入主结论**：
    它的统计口径（赛时）与 A/C 格（累计快照）不同，混进同一张主表会造成口径混淆。
    表 4 中单独成行，用于回答"若新题在赛后即获得赛时统计，能达到什么水平"。

    前置条件：需先运行 `python scripts/build_s1_in_contest.py`，并用
    `build_dataset.py --skip-crawl` 重建数据集（S1' 列才有值）。
    """
    log("阶段 s1：因子设计第五格 S1'（时间划分 + 赛时统计）")
    cond = C.CONDITIONS.get("S1")
    if cond is None:
        log("  [skip] config.CONDITIONS 里没有 S1，跳过")
        return pd.DataFrame()
    # 复用 stage_factors 的全套逻辑（同一划分、同一超参、同样落盘预测缓存），
    # 只把条件限定为 S1 并给文件名加后缀，避免与 A/B/C/D 的结果混淆。
    return stage_factors(ds, si, models, smoke, seeds, device=device,
                         conditions=("S1",), suffix="_s1")


def stage_verify(ds, si, smoke: bool, device=None) -> int:
    """★ 冒烟验证：全量重建后先跑一遍轻量检查，再决定是否启动完整实验。

    为什么需要它：完整实验要跑 6~9 小时（其中 BERT 占绝大部分）。
    若数据本身有问题（时间划分三段比例失真、某格样本量异常、S1' 列泄漏等），
    把 9 小时压在坏数据上代价很大。本阶段在**分钟级**内完成下列检查：

    1. **时间划分三段占比**是否回到约 65.5 / 7.5 / 27.0（大纲【十】实测值，
       容差 3 pp）；随机划分是否 80 / 10 / 10；
    2. **各格样本量**：A/B/C/D/S1' 五格的训练/验证/测试题数是否都够训练
       （验证集过小会让早停与分箱阈值不可靠）；
    3. **统计特征可得性**：冷启动格无统计列、普通格无 S1' 列（防静默泄漏）；
    4. **LightGBM 单模型四格**：确认整条链路能跑通并产出指标。

    任一硬检查失败即返回非零，**不继续**启动完整实验。
    """
    log(f"阶段 verify：全量重建后的轻量验证（不跑完整实验）")
    fails: list[str] = []

    # ---- 1. 划分占比 ----
    log("  [1/5] 划分三段占比")
    r_rand = si.ratios(C.SPLIT_RANDOM)
    r_temp = si.ratios(C.SPLIT_TEMPORAL)
    log(f"    随机：训练 {r_rand['train']:.1%} / 验证 {r_rand['valid']:.1%} "
        f"/ 测试 {r_rand['test']:.1%}（期望 80/10/10）")
    log(f"    时间：训练 {r_temp['train']:.1%} / 验证 {r_temp['valid']:.1%} "
        f"/ 测试 {r_temp['test']:.1%}（期望 65.5/7.5/27.0）")
    for part, exp, kind, got in (
            ("train", C.EXPECTED_TEMPORAL_RATIOS["train"], "时间", r_temp["train"]),
            ("valid", C.EXPECTED_TEMPORAL_RATIOS["valid"], "时间", r_temp["valid"]),
            ("test", C.EXPECTED_TEMPORAL_RATIOS["test"], "时间", r_temp["test"]),
            ("train", C.EXPECTED_RANDOM_RATIOS["train"], "随机", r_rand["train"])):
        diff_pp = abs(got - exp) * 100
        if diff_pp > C.RATIO_TOLERANCE_PP:
            fails.append(f"{kind}划分/{part} 占比 {got:.1%} 与期望 {exp:.1%} "
                         f"相差 {diff_pp:.1f} pp（容差 {C.RATIO_TOLERANCE_PP} pp）")

    # ---- 2. 各格样本量 ----
    log("  [2/5] 各格样本量")
    for key in ("A", "B", "C", "D", "S1"):
        cond = C.CONDITIONS.get(key)
        if cond is None:
            continue
        try:
            pkg = E.build_data_package(ds, si, cond,
                                       need_tfidf=(key == "A"),
                                       tfidf_max_features=5000)
        except Exception as e:
            fails.append(f"{key} 格构建失败: {type(e).__name__}: {str(e)[:100]}")
            continue
        n_tr, n_va, n_te = len(pkg.ds_train), len(pkg.ds_valid), len(pkg.ds_test)
        flag = ""
        if n_va < 50:
            flag = "  <- 验证集过小，早停与分箱阈值不可靠"
            fails.append(f"{key} 格验证集仅 {n_va} 题（<50）")
        if n_te < 50:
            flag = "  <- 测试集过小"
            fails.append(f"{key} 格测试集仅 {n_te} 题（<50）")
        log(f"    {key:3s} 训练 {n_tr:5d} / 验证 {n_va:5d} / 测试 {n_te:5d}{flag}")

    # ---- 3. 统计特征可得性 ----
    log("  [3/5] 统计特征可得性（防静默泄漏）")
    for key in ("A", "B", "C", "D", "S1"):
        cond = C.CONDITIONS.get(key)
        if cond is None:
            continue
        try:
            pkg = E.build_data_package(ds, si, cond)
            s1_cols = [c for c in pkg.ds_train.X.columns if c.endswith("__s1")]
            base = [c for c in C.STAT_FEATURES if c in pkg.ds_train.X.columns]
            E.assert_no_stat_leak(pkg.ds_train, cond.stat)
            if cond.stat == C.STAT_MASKED and base:
                fails.append(f"{key} 格为冷启动却含统计列 {base}")
            if key != "S1" and s1_cols:
                fails.append(f"{key} 格泄漏了 S1' 列 {s1_cols}")
            log(f"    {key:3s} 基础统计 {len(base)} 列，S1' 列 {len(s1_cols)} 个 -> OK")
        except Exception as e:
            fails.append(f"{key} 格特征可得性检查失败: {str(e)[:100]}")

    # ---- 4. LightGBM 单模型四格 ----
    log(f"  [4/5] LightGBM 单模型跑 A/B/C/D 四格（轻量链路检查）")
    try:
        params = E.load_hyperparams("lightgbm")
    except FileNotFoundError:
        params = {}
        log("    [note] 尚无 lightgbm 超参文件，用默认参数（调参阶段会重新搜索）")
    rows = []
    for key in ("A", "B", "C", "D"):
        cond = C.CONDITIONS[key]
        try:
            # ⚠ 这里刻意把 TF-IDF 维度压到 3000：verify 的目的是确认"链路能通、
            # 方向正确"，不是产出可用结果。全维 TF-IDF（20000）在四格上要额外
            # 花数分钟，而 verify 应当保持在分钟级——否则每次重建后都要多等，
            # 反而诱使人跳过它。真正实验仍用 build_data_package 的默认维度。
            pkg = E.build_data_package(ds, si, cond, need_tfidf=True,
                                       tfidf_max_features=3000)
            res = E.run_tabular(
                "lightgbm", pkg, params=params, seed=0,
                tag_columns=[c for c in pkg.ds_train.group_columns("tag")
                             if c.startswith(C.TAG_MULTIHOT_PREFIX)],
                allow_test_repeat=True)
            m = res.test_metrics
            log(f"    {key} 格：MAE {m['mae']:7.2f}  ±200 {m['hit200']:.3f}  "
                f"macroF1 {m['macro_f1']:.3f}  QWK {m['qwk']:.3f}  ρ {m['spearman']:.3f}"
                f"  （TF-IDF 限 3000 维，仅供链路检查）")
            rows.append((key, m["mae"]))
        except Exception as e:
            import traceback
            traceback.print_exc()
            fails.append(f"{key} 格 LightGBM 训练失败: {type(e).__name__}: {str(e)[:100]}")

    # 方向性检查：A 应优于 B（统计特征有效），B 应优于 D（时间漂移存在）
    d = dict(rows)
    if "A" in d and "B" in d and d["A"] >= d["B"]:
        fails.append(f"方向性异常：A 格 MAE {d['A']:.2f} 未优于 B 格 {d['B']:.2f}"
                     f"（统计特征应带来增益）")
    if "B" in d and "D" in d and d["B"] >= d["D"]:
        fails.append(f"方向性异常：B 格 MAE {d['B']:.2f} 未优于 D 格 {d['D']:.2f}"
                     f"（时间划分应带来泛化损失）")

    # ---- 5. S1' 格能否训练（第五格是新加的，最容易静默失败）----
    log("  [5/5] S1' 格（时间划分 + 赛时统计）链路检查")
    cond_s1 = C.CONDITIONS.get("S1")
    if cond_s1 is not None and not smoke:
        try:
            pkg_s1 = E.build_data_package(ds, si, cond_s1, need_tfidf=True,
                                          tfidf_max_features=3000)
            n_s1 = sum(1 for c in pkg_s1.ds_train.X.columns if c.endswith("__s1"))
            if n_s1 == 0:
                fails.append("S1' 格没有任何 __s1 列——赛时统计未接入，"
                             "请检查 build_s1_in_contest.py 与 build_dataset.py 是否跑过")
            else:
                res = E.run_tabular(
                    "lightgbm", pkg_s1, params=params, seed=0,
                    tag_columns=[c for c in pkg_s1.ds_train.group_columns("tag")
                                 if c.startswith(C.TAG_MULTIHOT_PREFIX)],
                    allow_test_repeat=True)
                m = res.test_metrics
                log(f"    S1' 格：MAE {m['mae']:7.2f}  ±200 {m['hit200']:.3f}  "
                    f"ρ {m['spearman']:.3f}（{n_s1} 个赛时统计列）")
                rows.append(("S1", m["mae"]))
        except Exception as e:
            fails.append(f"S1' 格链路检查失败: {type(e).__name__}: {str(e)[:120]}")
    else:
        log("    [skip] 无 S1 条件或处于 smoke 模式")

    log("")
    if fails:
        log(f"✗ 验证未通过（{len(fails)} 项）：")
        for f in fails:
            print(f"    ✗ {f}")
        print("\n  请先修复上述问题，**不要**启动完整实验（完整实验需 6~9 小时）。")
        return 1
    log("✓ 全部检查通过，可以启动完整实验：")
    print("    python scripts/run_all.py --from tune --seeds <...>")
    print("    （或 python scripts/run_all.py --skip-build，从 tune 一路跑到出图）")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="算法题难度预测实验运行器")
    ap.add_argument("--stage", default="all",
                    choices=["prepare", "tune", "factors", "s1", "size_control",
                             "data_efficiency", "ablation", "verify", "all"])
    ap.add_argument("--smoke", action="store_true",
                    help="用合成数据快速自测（不下载真实数据、不跑 BERT）")
    ap.add_argument("--models", nargs="*", default=None,
                    help="仅跑指定模型；默认全部主表模型")
    ap.add_argument("--seeds", nargs="*", type=int, default=None)
    ap.add_argument("--force", action="store_true", help="重调超参，覆盖已有 json")
    args = ap.parse_args()

    ensure_dirs()
    issues = C.validate()
    if issues:
        for i in issues:
            log(f"[口径自检失败] {i}")
        return 1

    ds = load_or_make_dataset(args.smoke)
    si = get_split_index(ds, args.smoke)
    log(f"数据集 {len(ds)} 题；划分索引就绪")

    models = args.models or list(C.MAIN_TABLE_MODELS)
    if args.smoke:
        # 合成自测：只跑快模型，BERT 单独手动验证
        models = [m for m in models if m not in ("bert",)]
    seeds = tuple(args.seeds) if args.seeds else ((0,) if args.smoke else C.SEEDS[:5])
    device = MD.pick_device()
    log(f"设备 {device}；种子 {seeds}；模型 {models}")

    # verify 单独处理：它自成一个"通过/不通过"的判定，且失败即中止
    if args.stage == "verify":
        return stage_verify(ds, si, args.smoke, device=device)

    if args.stage in ("prepare", "all"):
        stage_prepare(ds, si, args.smoke)
    if args.stage in ("tune", "all"):
        stage_tune(ds, si, models, args.smoke, device=device, force=args.force)
    if args.stage in ("factors", "all"):
        stage_factors(ds, si, models, args.smoke, seeds, device=device)
    if args.stage in ("s1", "all"):
        stage_s1(ds, si, models, args.smoke, seeds, device=device)
    if args.stage in ("size_control", "all"):
        stage_size_control(ds, si, args.smoke, device=device, seeds=seeds)
    if args.stage in ("data_efficiency", "all"):
        stage_data_efficiency(ds, si, args.smoke, device=device)
    if args.stage in ("ablation", "all"):
        stage_ablation(ds, si, args.smoke, device=device, seeds=seeds)

    log("完成。结果位于 outputs/results/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
