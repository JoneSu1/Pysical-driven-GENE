# -*- coding: utf-8 -*-
"""mech 分支改动的合成数据验证：
1. cnn: dropout 配置真正生效（getattr→.get 修复）；任意长度 ks/ds 可用；rf 计算正确
2. preprocess._balance_and_label: stratify_by='chrom' 分层平衡；默认行为与原版一致
3. preprocess.compile_training_data: 非法 target_transform 报错
4. trainer.Trainer: weight_decay 进 optimizer；lr_scheduler 选项正确构建；lr 记录进指标
运行：D:/Anacoda/envs/centri_c/python.exe tests/test_mech_changes.py
"""
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pandas as pd

# 测试不触碰 bigWig 读取；本地 CPU 环境无 pyBigWig，打桩绕过模块级 import
sys.modules.setdefault("pyBigWig", MagicMock())

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import torch
from deepISA.modeling.cnn import Conv
from deepISA.modeling.preprocess import (_balance_and_label,
                                         _subtract_window_proximity,
                                         compile_training_data)
from deepISA.modeling.trainer import Trainer


def test_cnn_dropout_and_rf():
    m = Conv(mode="dual", model_config={
        "seq_len": 600, "ks": [15, 9, 9, 9, 9], "cs": [64] * 5,
        "ds": [1, 2, 4, 8, 16], "dropout": 0.2})
    assert m.dropout.p == 0.2, f"dropout not honored: {m.dropout.p}"
    assert m.rf == 255, m.rf
    # 任意长度架构（rf 扩展只需 config，不需改代码）
    m6 = Conv(mode="dual", model_config={
        "seq_len": 600, "ks": [15, 9, 9, 9, 9, 9], "cs": [64] * 6,
        "ds": [1, 2, 4, 8, 16, 32], "dropout": 0.1})
    assert m6.rf == 511, m6.rf
    x = torch.randn(2, 4, 600)
    out = m6(x)
    assert out.shape == (2, 2), out.shape
    print("PASS cnn: dropout honored, rf 255/511, 6-layer forward OK")


def test_balance_stratified():
    rng = np.random.default_rng(0)
    n = 20000
    chroms = rng.choice(["chr1", "chr2", "chr19"], size=n, p=[0.5, 0.3, 0.2])
    df = pd.DataFrame({
        "chrom": chroms,
        "start": rng.integers(0, 10**7, n),
        "end": rng.integers(10**7 + 1, 10**7 + 600, n),
        # chr19 阳性率 45%，chr1 25% —— 偏斜但每层 neg 仍充足
        "target_class": [
            float(rng.random() < (0.45 if c == "chr19" else 0.25)) for c in chroms],
        "target_reg": rng.random(n),
    })
    neg_pool = df.copy()  # 不触发 extra_negs 分支

    out_s = _balance_and_label(df, neg_pool, 600, stratify_by="chrom")
    pos = out_s[out_s.target_class == 1.0]
    neg = out_s[out_s.target_class == 0.0]
    assert len(pos) == len(neg), (len(pos), len(neg))
    # 每层内部精确 1:1（池内 neg 充足时）
    for c in ["chr1", "chr2", "chr19"]:
        p_c = (pos.chrom == c).sum()
        n_c = (neg.chrom == c).sum()
        assert p_c == n_c, (c, p_c, n_c)

    # 短缺场景（chr19 pos>neg）：全局 1:1 仍成立，触发 top-up
    df2 = df.copy()
    df2.loc[(df2.chrom == "chr19"), "target_class"] = 1.0  # chr19 全阳性
    out2 = _balance_and_label(df2, neg_pool, 600, stratify_by="chrom")
    p2 = (out2.target_class == 1.0).sum()
    n2 = (out2.target_class == 0.0).sum()
    assert p2 == n2, (p2, n2)

    out_d = _balance_and_label(df, neg_pool, 600)  # 默认路径不变
    assert len(out_d) == 2 * (df.target_class == 1.0).sum()
    print("PASS balance: stratified per-chrom 1:1 OK; shortfall keeps global 1:1; "
          "default path unchanged")


def test_transform_validation():
    try:
        compile_training_data(pd.DataFrame(), fasta_path=None, out_dir=None,
                              target_transform="sqrt")
    except ValueError as e:
        assert "Unsupported target_transform" in str(e)
    else:
        raise AssertionError("invalid transform not rejected")
    print("PASS compile: invalid target_transform rejected")


def test_trainer_weight_decay_and_scheduler():
    m = Conv(mode="dual", model_config={
        "seq_len": 600, "ks": [15, 9, 9, 9, 9], "cs": [64] * 5,
        "ds": [1, 2, 4, 8, 16], "dropout": 0.1})
    with tempfile.TemporaryDirectory() as td:
        t = Trainer(m, "dual", None, None, None, torch.device("cpu"), td,
                    {"learning_rate": 3e-4, "weight_decay": 1e-4,
                     "lr_scheduler": "plateau", "epochs": 5})
        pg = t.optimizer.param_groups[0]
        assert pg["weight_decay"] == 1e-4 and abs(pg["lr"] - 3e-4) < 1e-9
        assert isinstance(t.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau)

        t2 = Trainer(m, "dual", None, None, None, torch.device("cpu"), td,
                     {"lr_scheduler": "cosine"})
        assert isinstance(t2.scheduler, torch.optim.lr_scheduler.CosineAnnealingLR)
        assert t2.optimizer.param_groups[0]["weight_decay"] == 0.0  # 默认不变

        t3 = Trainer(m, "dual", None, None, None, torch.device("cpu"), td, {})
        assert t3.scheduler is None  # 默认行为 = 原版
        try:
            Trainer(m, "dual", None, None, None, torch.device("cpu"), td,
                    {"lr_scheduler": "bogus"})
        except ValueError:
            pass
        else:
            raise AssertionError("bad scheduler not rejected")
    print("PASS trainer: weight_decay/plateau/cosine OK, defaults = upstream")


def test_val_split_options():
    # val_chrom 拒绝 chr2（与 test 冲突）
    try:
        compile_training_data(pd.DataFrame(), fasta_path=None, out_dir=None,
                              val_chrom="chr2")
    except ValueError as e:
        assert "chr2" in str(e)
    else:
        raise AssertionError("val_chrom=chr2 not rejected")

    # 重叠隔离带：val 窗口 [1000,1600)，exclusion 600 → [400,2200)
    val = pd.DataFrame({"chrom": ["chr1"], "start": [1000], "end": [1600]})
    train = pd.DataFrame({
        "chrom": ["chr1", "chr1", "chr1", "chr2"],
        "start": [0, 1500, 5000, 1000],      # 0: [0,600) 与 [400,2200) 重叠；1500 重叠；5000 不重叠
        "end":   [600, 2100, 5600, 1600],     # chr2 不在 val 的染色体上 → 保留
    })
    out = _subtract_window_proximity(train, val, 600)
    assert list(out["start"]) == [5000, 1000], out
    # exclusion=0 语义 = 不动（由 compile 里的分支保证，这里直接测 helper 边界）
    out0 = _subtract_window_proximity(train, val, 0)
    assert list(out0["start"]) == [0, 5000, 1000], out0  # [0,600) 与 [1000,1600) 不相接 → 保留
    print("PASS val split: chr2 rejected; ±600bp proximity exclusion exact")


def test_balance_skip_endtoend(tmp_path=None):
    """Scenario 1 (pre-quantified) + balance=False：_balance_and_label 不被调用，
    拆分/memmap 正常；chr7 作 val、chr2 作 test。"""
    import deepISA.modeling.preprocess as prep

    rng = np.random.default_rng(1)
    n = 600
    df = pd.DataFrame({
        "chrom": rng.choice(["chr1", "chr2", "chr7"], size=n),
        "start": rng.integers(0, 9000, n),
        "sig": rng.random(n) * 5,
    })
    df["end"] = df["start"] + 600
    df["target_reg"] = df["sig"]

    # 打桩：fasta 用随机序列字典；_balance_and_label 被调用即失败
    fake_fasta = {c: "".join(rng.choice(list("ACGT"), 20000))
                  for c in ["chr1", "chr2", "chr7"]}
    orig_load, orig_bal = prep.bf.load_fasta, prep._balance_and_label
    prep.bf.load_fasta = lambda *a, **k: fake_fasta
    def _boom(*a, **k):
        raise AssertionError("_balance_and_label must not be called when balance=False")
    prep._balance_and_label = _boom
    try:
        with tempfile.TemporaryDirectory() as td:
            out = prep.compile_training_data(
                df, fasta_path="fake", out_dir=td, seq_len=600,
                target_reg_col="target_reg", rc_aug=False,
                balance=False, val_chrom="chr7")
            assert len(out) == n
            for split, chroms in [("train", {"chr1"}), ("val", {"chr7"}),
                                  ("test", {"chr2"})]:
                meta = pd.read_json(Path(td) / split / "metadata.json",
                                    typ="series")
                assert int(meta["X"][0]) > 0
            # 再确认默认 balance=True 仍会调用（用原版函数放回并打桩计数）
            calls = {"n": 0}
            def _count(*a, **k):
                calls["n"] += 1
                return a[0]
            prep._balance_and_label = _count
            prep.compile_training_data(
                df, fasta_path="fake", out_dir=td, seq_len=600,
                target_reg_col="target_reg", rc_aug=False)
            assert calls["n"] == 1
    finally:
        prep.bf.load_fasta = orig_load
        prep._balance_and_label = orig_bal
    print("PASS balance=False: end-to-end compile, balance skipped; "
          "default still balances; chr7/chr2 splits OK")


if __name__ == "__main__":
    test_cnn_dropout_and_rf()
    test_balance_stratified()
    test_transform_validation()
    test_trainer_weight_decay_and_scheduler()
    test_val_split_options()
    test_balance_skip_endtoend()
    print("\nALL MECH TESTS PASSED")
