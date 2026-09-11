# deepISA mech 分支说明（FORK_NOTES）

> 基线：upstream `anderssonlab/deepISA` @ `34a52cc`（2026-05-26，v1 全部模型训练所用代码）。
> 本分支（main）= 最小差量 fork：默认参数下行为与上游逐位一致，所有新增能力都是 config 开关。
> 许可证注意：上游无 LICENSE 文件，公开 push 前建议先请上游补许可证（或保持 private）。

## 与上游的差异（git diff 34a52cc..main 可见全部）

### 1. `cnn.py` — dropout 配置修复（bug fix）
上游 `getattr(model_config, 'dropout', 0.1)` 对 dict 永远回退 0.1，配置的 dropout 从不生效
（上游后来在 `f527785` 也修了同一处，修法相同）。
影响：**2026-09-10 converged run2 设了 dropout=0.2 但实际按 0.1 训练**——跑数字时以 0.1 为准。

### 2. `preprocess.py` — 两个新开关（compile_training_data）
- `target_transform="log1p"`：对区域信号与 P99 噪声阈值在同一空间做 log1p
  （单调变换 → target_class 标签与默认路径逐位一致；raw 标度 Pearson 不可跨标度直接比）。
- `balance_stratify="chrom"`：负样本按染色体配额下采样（quota 取阳性的 chr 组成），
  每层 pos:neg≈1:1；某 chr 阴性不足时全局随机补齐（真实 1kPa 数据短缺 ~3,011/88k，3.4%），
  全局 1:1 始终成立。默认 `None` = 上游行为。

### 3. `trainer.py` — 训练循环增强（trainer_config 新键，默认全关）
- `weight_decay`（默认 0.0）进 Adam。
- `lr_scheduler`: `None`（默认）| `"plateau"`（ReduceLROnPlateau mode=max，跟 val Pearson，
  factor 0.5 patience 5）| `"cosine"`（CosineAnnealingLR T_max=epochs）。
- `metrics.csv` 每轮多记一列 `lr`，调度行为可审计。

### 4. `quickstart.py` — `QuickStart.train()` 透传 `target_transform` / `balance_stratify` / `val_chrom` / `val_exclusion_bp` / `balance`。

### 5. `preprocess.py` — val/test 拆分增强（commit d7b526a）
- `val_chrom="chr7"`（等）：整条染色体作验证集——early stopping/模型选择与 chr2 test
  一样测跨染色体泛化（原版 val = train 同染色体随机 15%，只测插值）。
  `val_chrom="chr2"` 被拒绝（与 test 冲突）。**不推荐** chr2 前后半段切 val/test：
  同染色体 Mb 尺度染色质强相关，val 会通过模型选择间接偏向 chr2，稀释 test 的泛化语义。
- `val_exclusion_bp=600`（仅随机 val 模式）：剔除与任何 val 窗口距离 <N bp 的 train 窗口，
  堵上 600bp resize cCRE 的窗口重叠泄漏；merge+searchsorted 精确实现。默认 0 = 上游。

### 6. `preprocess.py`/`quickstart.py` — `balance=False`（commit c231572）
跳过 1:1 平衡：发现 pos-only 臂的"阴性"实际 ~87k 个是背景填充零信号（extra_negs 分支），
占一半 batch、使 BCE 头退化、稀释回归梯度。纯阳性臂用 `balance=False` + `mode="regression"`。
**注意**：v1 的 pos-only Pearson 0.844 是在"阳性+背景"混合集上算的，含分离成分、偏高；
纯阳性臂的 within-positive Pearson 预期更低但更真实，两者不可直接比。

### 8. 可复现种子（commit 84fbffa）
此前只有数据拆分/抽样有 random_state=42，模型初始化、batch 洗牌、dropout 无种子——
重跑同 notebook 得到不同模型。现在：utils.set_seed(seed)（random/np/torch/cuda/cudnn deterministic）；
define_model(seed=) 固定权重初始化；trainer_config["seed"] 在 train() 开始时固定洗牌/dropout 流。
同 seed 两次初始化权重逐位一致（已测）。注意 cudnn.deterministic 会损失少量训练速度。
**已存模型（v1、run2、v3 首发、v3b）都是无种子训练的，不可逐位复现；v3c 起带种子。**

### 8. `tests/test_mech_changes.py` — 合成数据验证
dropout 生效、rf 255/511（6 层）、6 层前向、分层平衡精确 1:1 + 短缺保全局 1:1、
默认路径与上游一致、非法参数拒绝、val 拆分/隔离带、balance=False 端到端。
运行（本地 CPU 即可，pyBigWig 打桩）：

```
python tests/test_mech_changes.py   # 期望 7× PASS
```

## 架构说明：感受野扩展不需要改代码

`Conv` 的 ks/cs/ds 是任意长度 config 驱动的。默认 5 层 rf=255 < 600 输入窗。
扩展只改 model_config，例如 6 层 `ds=[1,2,4,8,16,32]` → rf=511
（已在测试中验证前向可用）。注意 `model_config.json` 随模型落盘，下游加载要按它来。

## notebook 迁移（push 到个人 GitHub 后）

converged notebook 的 Cell 3 两行替换即可：

```python
!git clone https://github.com/<你的用户名>/deepISA.git {DEEPISA_DIR}
!cd {DEEPISA_DIR} && git checkout main   # 或打 tag 后 checkout tag
```

之后 Cell 7.5 补丁格整个删除，训练调用改为：

```python
qs.train(trainer_config=trainer_config, bw_paths=bw_paths,
         rc_aug=True, target_transform="log1p", balance_stratify="chrom")
```

## 合并上游更新的纪律

上游在 ISA/discover 模块活跃演进；合并前先 `git diff` 审建模/预处理三个文件，
本分支已动的文件合并冲突时以本分支语义为准（config 开关向后兼容）。
