"""深度侧模型与训练循环（大纲三·（二）（三））。

主表深度侧 4 项：MLP / TextCNN / BERT / 融合模型；BiLSTM+Attention 入附表。

★ 本机显存约束（实测，非估算）
------------------------------
RTX 5070 Laptop 总显存 7.96 GB，但 Windows 桌面/浏览器常驻占用约 5.4 GB，
**实际可用仅约 2.6 GB**。实测 bert-base-uncased：
  max_length=256, batch=16 → 峰值 3.71 GB（需与桌面争抢，但可跑）
  max_length=256, batch=32 → 峰值 6.07 GB（会触发换页，步时 0.74s→3.76s，慢 5 倍）
  max_length=512, batch=16 → 峰值 6.06 GB（不可用）
因此训练器**默认开启 fp16 混合精度**并支持梯度累积——
这不是性能优化，而是本机上能否跑通的必要条件。论文的
"文本长度"进入引言/表 1 时，也应说明 max_length 的选取依据。

★ 训练口径
----------
早停与选模**只在验证集**上做；测试集只在最后预测一次。
早停监控 MAE（大纲的回归主指标），而不是 loss。
"""

from __future__ import annotations

import copy
import math
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from . import config as C

# ================================================================ 设备


def pick_device(prefer: str = "cuda") -> "Any":
    """选择设备，并对 Blackwell（sm_120）做内核可用性自检。

    进入本函数时顺手打开全局加速开关（TF32 / cuDNN benchmark / 显存分配策略），
    必须在第一次 CUDA 调用之前设置，因此放在设备选择的入口处。
    """
    import torch

    from . import accel as A

    A.configure_torch()
    if prefer == "cuda" and torch.cuda.is_available():
        from .sysinfo import assert_gpu_supported

        ok, msg = assert_gpu_supported()
        if ok:
            return torch.device("cuda")
        # sm_120 不在编译架构里时，硬上会报 kernel 错误；此处明确告警并退回 CPU
        print(f"[warn] GPU 自检未通过，回退 CPU: {msg}")
    return torch.device("cpu")


def vram_report() -> dict[str, float]:
    import torch

    if not torch.cuda.is_available():
        return {}
    free, total = torch.cuda.mem_get_info()
    return {
        "free_gb": free / 1024 ** 3,
        "total_gb": total / 1024 ** 3,
        "used_gb": (total - free) / 1024 ** 3,
        "peak_alloc_gb": torch.cuda.max_memory_allocated() / 1024 ** 3,
        "peak_reserved_gb": torch.cuda.max_memory_reserved() / 1024 ** 3,
    }


# ================================================================ 批次容器


@dataclass
class Batch:
    """统一批次：结构化特征 + 文本 id 序列 + 标注。"""

    struct: np.ndarray | None = None      # (B, d_struct) 允许含 NaN
    token_ids: np.ndarray | None = None   # (B, L) 文本 id（BERT 或 SimpleVocab）
    attention_mask: np.ndarray | None = None
    token_type_ids: np.ndarray | None = None
    y: np.ndarray | None = None           # (B,) rating

    def __len__(self) -> int:
        for v in (self.y, self.struct, self.token_ids):
            if v is not None:
                return len(v)
        return 0


def make_loader(batch: Batch, batch_size: int, shuffle: bool,
                seed: int = 0, drop_last: bool = False):
    """把**单个** Batch（全量数组）切成 tensor 批次。

    不直接依赖 DataLoader 的默认 collate——Batch 里有可选字段（struct 可能为
    None），必须按"该条件实际存在哪些字段"动态建列：冷启动格没有 struct，
    若强行堆叠 NaN 数组会平白多出一个输入通道。
    """
    import torch
    from torch.utils.data import DataLoader

    n = len(batch)
    if n == 0:
        raise ValueError("batch 为空")

    cols: dict[str, Any] = {}
    for key in ("struct", "token_ids", "attention_mask", "token_type_ids"):
        v = getattr(batch, key)
        if v is None:
            continue
        is_long = key in ("token_ids", "attention_mask", "token_type_ids")
        cols[key] = torch.as_tensor(
            np.asarray(v), dtype=torch.long if is_long else torch.float32
        )
    if batch.y is None:
        raise ValueError("Batch.y 不能为空")
    cols["y"] = torch.as_tensor(np.asarray(batch.y), dtype=torch.float32)
    order = list(cols.keys())

    class _DS(torch.utils.data.Dataset):
        def __len__(self):
            return n

        def __getitem__(self, i):
            return {k: cols[k][i] for k in order}

    return DataLoader(_DS(), batch_size=batch_size, shuffle=shuffle,
                      drop_last=drop_last,
                      generator=torch.Generator().manual_seed(seed) if shuffle else None,
                      num_workers=0,
                      # 锁页内存让 H2D 拷贝走 DMA、可与计算重叠；8 GB 显存下
                      # 这一步的收益主要体现在 BERT 的 token/label 批次上
                      pin_memory=torch.cuda.is_available())


# ================================================================ 网络

def build_mlp(d_struct: int, hidden: Sequence[int] = (256, 128), dropout: float = 0.3,
              in_mean: np.ndarray | None = None, in_std: np.ndarray | None = None):
    """MLP：结构化特征通道基准（只看标签/约束/统计，不看题面文本）。

    ★ 输入标准化是必需的，不是可选项：本任务结构化特征跨度极大
    （solved_count 可达 1e6，多测组数上界 1e4，交互题标志是 0/1），
    不做标准化时即使 lr=3e-3 也会在第一个 batch 就梯度爆炸、loss 变 NaN
    （实测）。`in_mean/in_std` **只能由训练集算出**，后传入缓存。

    这里包一层适配器把关键字参数吸收掉——`nn.Sequential` 只接受位置参数，
    而统一训练循环总是以 `model(token_ids=..., struct=...)` 形式调用。
    """
    import torch
    import torch.nn as nn

    layers: list[nn.Module] = []
    prev = d_struct
    for h in hidden:
        layers += [nn.Linear(prev, h), nn.ReLU(), nn.Dropout(dropout)]
        prev = h
    layers.append(nn.Linear(prev, 1))
    seq = nn.Sequential(*layers)

    class MLP(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = seq
            mu = np.zeros(d_struct, dtype=np.float32) if in_mean is None \
                else np.asarray(in_mean, dtype=np.float32)
            sd = np.ones(d_struct, dtype=np.float32) if in_std is None \
                else np.asarray(in_std, dtype=np.float32)
            sd = np.where(np.abs(sd) < 1e-8, 1.0, sd).astype(np.float32)
            self.register_buffer("in_mean", torch.as_tensor(mu))
            self.register_buffer("in_std", torch.as_tensor(sd))

        def forward(self, struct=None, **kw):
            if struct is None:
                raise ValueError("MLP 需要 struct 输入")
            # NaN 表示该特征不可得（冷启动），置 0 而非填补训练集统计量
            x = torch.nan_to_num(struct.float(), nan=0.0)
            x = (x - self.in_mean) / self.in_std     # 标准化（参数来自训练集）
            return self.net(x).squeeze(-1)

    return MLP()


def build_textcnn(vocab_size: int, embed_dim: int = 128,
                  kernel_sizes: Sequence[int] = (2, 3, 4, 5),
                  n_filters: int = 100, dropout: float = 0.5,
                  pad_idx: int = 0):
    """TextCNN（Kim, 2014）：多尺度卷积核 2/3/4/5 + Dropout。

    中文直觉解释（该刊读者为高职师生，正文宜如此写）：
    用 2~5 个词的"滑动窗口"去扫题面，每个窗口学一种"局部套路"的迹象
    （如"minimize the maximum"暗示二分答案、"number of ways"暗示计数/DP），
    再把各尺度最强的信号拼起来判断难度。
    """
    import torch
    import torch.nn as nn

    class TextCNN(nn.Module):
        def __init__(self):
            super().__init__()
            self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
            self.convs = nn.ModuleList([
                nn.Conv1d(embed_dim, n_filters, k) for k in kernel_sizes
            ])
            self.drop = nn.Dropout(dropout)
            self.fc = nn.Linear(n_filters * len(kernel_sizes), 1)

        def forward(self, token_ids, attention_mask=None, **kw):
            x = self.embed(token_ids)                    # (B, L, E)
            x = x.transpose(1, 2)                        # (B, E, L)
            feats = []
            for conv in self.convs:
                h = torch.relu(conv(x))                  # (B, F, L-k+1)
                h = torch.max(h, dim=2).values           # 全局最大池化
                feats.append(h)
            h = self.drop(torch.cat(feats, dim=1))
            return self.fc(h).squeeze(-1)

    return TextCNN()


def build_bilstm_attn(vocab_size: int, embed_dim: int = 128, hidden: int = 128,
                      dropout: float = 0.5, pad_idx: int = 0):
    """BiLSTM + Attention（**附表项**，不进主表）。

    作为序列建模路线的代表保留，但增量信息有限——大纲已明确其定位，
    正文不得因"跑了就写"而把它塞进主表。
    """
    import torch
    import torch.nn as nn

    class BiLSTMAttn(nn.Module):
        def __init__(self):
            super().__init__()
            self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
            self.lstm = nn.LSTM(embed_dim, hidden, batch_first=True,
                                bidirectional=True)
            self.att = nn.Linear(hidden * 2, 1)
            self.drop = nn.Dropout(dropout)
            self.fc = nn.Linear(hidden * 2, 1)

        def forward(self, token_ids, attention_mask=None, **kw):
            x = self.embed(token_ids)
            h, _ = self.lstm(x)                          # (B, L, 2H)
            scores = self.att(h).squeeze(-1)             # (B, L)
            if attention_mask is not None:
                scores = scores.masked_fill(attention_mask == 0, -1e9)
            w = torch.softmax(scores, dim=1).unsqueeze(-1)
            ctx = (h * w).sum(dim=1)
            return self.fc(self.drop(ctx)).squeeze(-1)

    return BiLSTMAttn()


def build_bert_regressor(model_name: str = "bert-base-uncased",
                         dropout: float = 0.1, freeze_embeddings: bool = False):
    """BERT 微调：[CLS] 接**回归头**（大纲三·（二））。

    注意：不使用 BertForSequenceClassification——那是分类头，
    会诱导"独立分类训练"，违反"分类指标由回归输出分箱派生"的红线。
    """
    import torch
    import torch.nn as nn
    from transformers import AutoConfig, AutoModel

    class BertRegressor(nn.Module):
        def __init__(self):
            super().__init__()
            cfg = AutoConfig.from_pretrained(model_name)
            self.bert = AutoModel.from_pretrained(model_name, config=cfg)
            if freeze_embeddings:
                self.bert.embeddings.requires_grad_(False)
            self.drop = nn.Dropout(dropout)
            self.head = nn.Linear(cfg.hidden_size, 1)

        def forward(self, token_ids, attention_mask=None, token_type_ids=None, **kw):
            out = self.bert(
                input_ids=token_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
            )
            cls = out.last_hidden_state[:, 0]            # [CLS]
            return self.head(self.drop(cls)).squeeze(-1)

        @property
        def hidden_size(self) -> int:
            return int(self.bert.config.hidden_size)

    return BertRegressor()


def build_fusion(model_name: str = "bert-base-uncased", d_struct: int = 0,
                 struct_hidden: int = 128, fusion_hidden: Sequence[int] = (256, 128),
                 dropout: float = 0.3, freeze_bert: bool = False,
                 in_mean: np.ndarray | None = None, in_std: np.ndarray | None = None):
    """双通道融合模型（大纲三·（三））——**定位为"基线级融合"，不宣称结构创新**。

    结构：BERT 文本向量 ⊕ 结构化特征（经全连接映射同维）→ 拼接 → MLP → **回归输出**。

    为什么融合：文本反映"考什么"，统计反映"考倒谁"。
    边界（必须写进正文）：融合收益随统计特征可得性下降而**收窄**，
    但**纯冷启动下并不归零**——结构化通道仍含算法标签与约束规模两组完全可得特征，
    故保留**少量**残余增益，该数值必须在 B/D 格中如实报告。

    结构化通道同样做输入标准化（参数只来自训练集），理由见 build_mlp 的注释。
    """
    import torch
    import torch.nn as nn
    from transformers import AutoConfig, AutoModel

    class Fusion(nn.Module):
        def __init__(self):
            super().__init__()
            cfg = AutoConfig.from_pretrained(model_name)
            self.bert = AutoModel.from_pretrained(model_name, config=cfg)
            if freeze_bert:
                for p in self.bert.parameters():
                    p.requires_grad_(False)
            h = int(cfg.hidden_size)
            mu = np.zeros(d_struct, dtype=np.float32) if in_mean is None \
                else np.asarray(in_mean, dtype=np.float32)
            sd = np.ones(d_struct, dtype=np.float32) if in_std is None \
                else np.asarray(in_std, dtype=np.float32)
            sd = np.where(np.abs(sd) < 1e-8, 1.0, sd).astype(np.float32)
            self.register_buffer("in_mean", torch.as_tensor(mu))
            self.register_buffer("in_std", torch.as_tensor(sd))
            self.struct_enc = nn.Sequential(
                nn.Linear(d_struct, struct_hidden), nn.ReLU(), nn.Dropout(dropout),
                nn.Linear(struct_hidden, h), nn.ReLU(),
            )
            layers: list[nn.Module] = []
            prev = h * 2
            for fh in fusion_hidden:
                layers += [nn.Linear(prev, fh), nn.ReLU(), nn.Dropout(dropout)]
                prev = fh
            layers.append(nn.Linear(prev, 1))
            self.head = nn.Sequential(*layers)

        def forward(self, token_ids, attention_mask=None, token_type_ids=None,
                    struct=None, **kw):
            out = self.bert(
                input_ids=token_ids, attention_mask=attention_mask,
                token_type_ids=token_type_ids,
            )
            txt = out.last_hidden_state[:, 0]
            # NaN 在结构化通道里表示"该特征不可得"（冷启动），先置 0 再标准化
            s = torch.nan_to_num(struct.float(), nan=0.0)
            s = (s - self.in_mean) / self.in_std
            st = self.struct_enc(s)
            return self.head(torch.cat([txt, st], dim=1)).squeeze(-1)

    return Fusion()


# ================================================================ 训练器


@dataclass
class TargetScaler:
    """目标值（rating）的仿射标准化。

    为什么必需，而不是可选
    ----------------------
    回归头初始化输出 ≈ 0，而 rating 在 800~3500。不标准化时首个 batch 的
    MSE 就高达数百万，模型要先花若干 epoch 把输出 bias "爬"到正确量级：
    实测 3 个 epoch 后验证 MAE 仍有 1920（≈ 直接预测常数）。

    对 BERT 这尤其致命——lr=2e-5、只微调 2~3 个 epoch（大纲四·（一）的预算），
    完全来不及完成这段爬升，结果会是"BERT 明显很差"，
    而那不是模型的真实能力，是训练配方的问题。

    因此所有深度模型统一在**标准化空间**训练，预测时逆变换回 rating 空间。
    这只影响优化的数值条件，不改变任何指标定义：对外输出的仍是 rating 点。
    """

    mean: float = 0.0
    std: float = 1.0

    @classmethod
    def fit(cls, y: np.ndarray) -> "TargetScaler":
        y = np.asarray(y, dtype=np.float64)
        return cls(mean=float(y.mean()), std=float(y.std()) or 1.0)

    def forward(self, y: np.ndarray) -> np.ndarray:
        return (np.asarray(y, dtype=np.float64) - self.mean) / self.std

    def forward_torch(self, y):
        """张量版标准化，**留在设备上算**。

        训练循环里若用 `y.cpu().numpy()` 再过 numpy，每个 step 都会强制一次
        设备同步，H2D/D2H 拷贝与计算无法重叠，BERT 的 GPU 利用率明显掉下来。
        目标值只是仿射变换，直接在设备上做即可。
        """
        return (y - self.mean) / self.std

    def inverse(self, z: np.ndarray) -> np.ndarray:
        return np.asarray(z, dtype=np.float64) * self.std + self.mean


@dataclass
class TrainConfig:
    """深度模型的训练超参（超参搜索的对象）。"""

    lr: float = 2e-5
    batch_size: int = 16
    epochs: int = 3
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    grad_accum: int = 1
    max_grad_norm: float = 1.0
    fp16: bool = True
    patience: int = 2
    dropout: float = 0.3
    embed_dim: int = 128
    max_length: int = 256
    #: TextCNN / BiLSTM 的序列长度上限
    seq_length: int = 400
    #: 是否做目标标准化（强烈建议开启，见 TargetScaler 的说明）
    scale_target: bool = True

    def as_dict(self) -> dict[str, Any]:
        return {k: getattr(self, k) for k in self.__dataclass_fields__}


@dataclass
class TrainHistory:
    epoch: list[int] = field(default_factory=list)
    train_loss: list[float] = field(default_factory=list)
    valid_mae: list[float] = field(default_factory=list)
    valid_rmse_for_earlystop_only: list[float] = field(default_factory=list)
    seconds: list[float] = field(default_factory=list)
    best_epoch: int = -1
    best_valid_mae: float = float("inf")
    stopped_early: bool = False
    peak_vram_gb: float = 0.0
    total_seconds: float = 0.0

    #: ⚠ 早停只许看 valid_mae。RMSE 仅内部记录，**不进入论文指标**
    #: （大纲三·（五）已删 RMSE/R²）。


def train_torch_model(
    model,
    train_loader,
    valid_loader,
    cfg: TrainConfig,
    device,
    valid_mae_fn: Callable[[np.ndarray, np.ndarray], float] | None = None,
    log: Callable[[str], None] | None = None,
    scaler_target: TargetScaler | None = None,
) -> tuple[Any, TrainHistory, np.ndarray]:
    """训练一个 torch 模型。返回 (best_model, history, valid_pred)。

    铁律：
      - 早停与选模**只看验证集 MAE**（且只在验证集上算）；
      - 恢复的是**验证集最优**的那个 checkpoint，不是最后一个 epoch；
      - 测试集不在此函数中出现；
      - 所有指标计算都在 **rating 空间**进行：内部若做了目标标准化，
        返回前一律逆变换回去。`history.valid_mae` 也记录 rating 空间的值。
    """
    import torch
    import torch.nn as nn

    if valid_mae_fn is None:
        valid_mae_fn = lambda a, b: float(np.mean(np.abs(np.asarray(a) - np.asarray(b))))

    use_scale = bool(cfg.scale_target and scaler_target is not None)
    sc = scaler_target if use_scale else TargetScaler(0.0, 1.0)

    model = model.to(device)
    opt = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=cfg.lr, weight_decay=cfg.weight_decay,
    )
    total_steps = max(1, len(train_loader) * cfg.epochs // max(1, cfg.grad_accum))
    warmup = max(1, int(total_steps * cfg.warmup_ratio))

    def lr_at(step: int) -> float:
        if step < warmup:
            return step / max(1, warmup)
        prog = (step - warmup) / max(1, total_steps - warmup)
        return max(0.0, 0.5 * (1 + math.cos(math.pi * min(1.0, prog))))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_at)
    use_amp = bool(cfg.fp16 and device.type == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    loss_fn = nn.SmoothL1Loss(beta=100.0)   # Huber：对 800 下钳位造成的标签噪声更稳

    history = TrainHistory()
    best_state = copy.deepcopy(model.state_dict())
    best_mae = float("inf")
    best_epoch = -1
    bad_epochs = 0
    t_start = time.time()

    for epoch in range(cfg.epochs):
        model.train()
        t0 = time.time()
        running, n_seen = 0.0, 0
        opt.zero_grad(set_to_none=True)

        for step, batch in enumerate(train_loader):
            ids = batch.get("token_ids")
            am = batch.get("attention_mask")
            tt = batch.get("token_type_ids")
            st = batch.get("struct")
            y = batch["y"].to(device, non_blocking=True)

            with torch.amp.autocast("cuda", enabled=use_amp):
                pred = _forward(model, ids, am, tt, st, device)
                # 目标标准化在设备上完成，不做 cpu()/numpy() 往返（避免每步同步）
                target = sc.forward_torch(y).to(dtype=pred.dtype)
                loss = loss_fn(pred, target) / cfg.grad_accum

            scaler.scale(loss).backward()
            if (step + 1) % cfg.grad_accum == 0:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad],
                    cfg.max_grad_norm,
                )
                scaler.step(opt)
                scaler.update()
                opt.zero_grad(set_to_none=True)
                sched.step()

            running += float(loss.item()) * cfg.grad_accum * len(y)
            n_seen += len(y)

        # ---- 验证（只用于早停/选模）----
        # 逆变换回 rating 空间再算 MAE，保证早停判据与最终报告口径一致
        valid_pred = predict_torch_model(model, valid_loader, device,
                                        amp=use_amp, scaler_target=sc)
        valid_true = collect_targets(valid_loader)
        vmae = float(valid_mae_fn(valid_true, valid_pred))

        history.epoch.append(epoch + 1)
        history.train_loss.append(running / max(1, n_seen))
        history.valid_mae.append(vmae)
        history.valid_rmse_for_earlystop_only.append(
            float(np.sqrt(np.mean((valid_true - valid_pred) ** 2)))
        )
        history.seconds.append(time.time() - t0)

        if log:
            log(f"    epoch {epoch+1}/{cfg.epochs} loss={history.train_loss[-1]:.2f} "
                f"valid_MAE={vmae:.2f} ({history.seconds[-1]:.1f}s)")

        if vmae < best_mae - 1e-6:
            best_mae, best_epoch, bad_epochs = vmae, epoch + 1, 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            bad_epochs += 1
            if bad_epochs >= cfg.patience:
                history.stopped_early = True
                if log:
                    log(f"    早停于 epoch {epoch+1}（验证 MAE 连续 {cfg.patience} 轮未改善）")
                break

    model.load_state_dict(best_state)
    history.best_epoch = best_epoch
    history.best_valid_mae = best_mae
    history.total_seconds = time.time() - t_start
    if device.type == "cuda":
        history.peak_vram_gb = torch.cuda.max_memory_allocated() / 1024 ** 3

    final_valid = predict_torch_model(model, valid_loader, device,
                                     amp=use_amp, scaler_target=sc)
    return model, history, final_valid


def _forward(model, ids, am, tt, st, device):
    kwargs: dict[str, Any] = {}
    if ids is not None:
        kwargs["token_ids"] = ids.to(device, non_blocking=True)
    if am is not None:
        kwargs["attention_mask"] = am.to(device, non_blocking=True)
    if tt is not None:
        kwargs["token_type_ids"] = tt.to(device, non_blocking=True)
    if st is not None:
        kwargs["struct"] = st.to(device, non_blocking=True)
    return model(**kwargs)


def predict_torch_model(model, loader, device, amp: bool = True,
                        scaler_target: "TargetScaler | None" = None) -> np.ndarray:
    """推理。测试集只应调用一次（由实验层保证）。

    `scaler_target` 给出时，输出会**逆变换回 rating 空间**——
    对外一律返回 rating 点，指标层不需要知道标准化这件事。
    """
    import torch

    model.eval()
    preds: list[np.ndarray] = []
    with torch.no_grad():
        for batch in loader:
            with torch.amp.autocast("cuda", enabled=bool(amp and device.type == "cuda")):
                p = _forward(model, batch.get("token_ids"), batch.get("attention_mask"),
                             batch.get("token_type_ids"), batch.get("struct"), device)
            preds.append(p.float().detach().cpu().numpy())
    out = np.concatenate(preds) if preds else np.empty(0, dtype=np.float64)
    if scaler_target is not None:
        out = scaler_target.inverse(out)
    return out


def collect_targets(loader) -> np.ndarray:
    ys: list[np.ndarray] = []
    for batch in loader:
        ys.append(batch["y"].numpy())
    return np.concatenate(ys) if ys else np.empty(0, dtype=np.float64)


# ================================================================ 超参空间

#: 大纲四·（一）：TextCNN/MLP 小网格 10～15 组；BERT 小网格 6～12 组（学习率 × epoch）
DEEP_SEARCH_SPACE: dict[str, dict[str, Sequence[Any]]] = {
    "mlp": {
        "lr": [1e-3, 3e-3, 1e-2],
        "batch_size": [32, 64],
        "dropout": [0.1, 0.3, 0.5],
        "epochs": [30, 60],
    },
    "textcnn": {
        "lr": [3e-4, 1e-3, 3e-3],
        "batch_size": [16, 32],
        "dropout": [0.3, 0.5],
        "embed_dim": [100, 128],
        "epochs": [5, 8, 12],
    },
    # BERT：只搜学习率 × epoch（网格需窄，微调成本高）
    "bert": {
        "lr": [1e-5, 2e-5, 3e-5],
        "epochs": [2, 3, 4],
    },
    "fusion": {
        "lr": [1e-5, 2e-5, 3e-5],
        "epochs": [2, 3, 4],
    },
    "bilstm_attn": {
        "lr": [3e-4, 1e-3],
        "batch_size": [16, 32],
        "dropout": [0.3, 0.5],
        "epochs": [5, 8],
    },
}

#: 本机显存安全档位：超过该 batch×max_length 组合会触发换页（实测结论）
VRAM_SAFE_HINT: dict[str, str] = {
    "max_length_256_batch_16": "峰值 3.71 GB —— 可用（需与桌面争抢）",
    "max_length_256_batch_32": "峰值 6.07 GB —— 会换页，步时慢 5 倍，不建议",
    "max_length_512_batch_16": "峰值 6.06 GB —— 本机不可用",
}
