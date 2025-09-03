# requirements: pandas, numpy, torch
import os, json, random
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler


# ===== imbalance & augmentation toggles =====
ENABLE_SAMPLER = True   # 소수 클래스(1) 업샘플링
ENABLE_AUG     = True   # 학습셋만 가벼운 증강

# =============================
# PyCharm 원클릭 실행용 설정
# =============================
CSV_PATH   = "uguide_data.csv"
BATCH_SIZE = 128
EPOCHS     = 30
LR         = 1e-3
VALID_RATIO= 0.2
SEED       = 42
THRESH     = 0.5

# -----------------------------
# Utils
# -----------------------------
def set_seed(seed=42):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def group_train_valid_split(df, group_col="session_id", valid_ratio=0.2, seed=42):
    """
    - 세션이 2개 이상: 세션 단위 분할(누수 방지), 최소 1세션은 train/valid에 각각 존재
    - 세션이 1개 이하: 행(row) 단위 랜덤 분할, 최소 1샘플은 train/valid에 각각 존재(데이터가 2행 이상인 경우)
    """
    rng = np.random.RandomState(seed)
    df = df.reset_index(drop=True)

    sessions = df[group_col].dropna().unique().tolist()

    # === 케이스 1) 세션이 1개 이하: 행 단위 분할
    if len(sessions) <= 1:
        n = len(df)
        if n < 2:
            # 표본이 1개면 valid를 만들 수 없으니 전량 train
            return df.copy(), df.iloc[0:0].copy()

        n_valid = int(round(n * valid_ratio))
        # 최소 1개는 valid로, 최소 1개는 train으로 남기기
        n_valid = max(1, min(n - 1, n_valid))

        idx = np.arange(n)
        rng.shuffle(idx)
        valid_idx = set(idx[:n_valid])

        df_valid = df.iloc[list(valid_idx)].reset_index(drop=True)
        df_train = df.iloc[[i for i in idx if i not in valid_idx]].reset_index(drop=True)
        return df_train, df_valid

    # === 케이스 2) 세션이 2개 이상: 세션 단위 분할
    rng.shuffle(sessions)
    n_valid = int(round(len(sessions) * valid_ratio))
    # 최소 1세션은 train/valid에 남기기
    n_valid = max(1, min(len(sessions) - 1, n_valid))

    valid_g = set(sessions[:n_valid])
    is_valid = df[group_col].isin(valid_g)

    df_train = df[~is_valid].reset_index(drop=True)
    df_valid = df[is_valid].reset_index(drop=True)
    return df_train, df_valid


# -----------------------------
# Dataset
# -----------------------------
SEQ_T = 5

def _pick_col(df, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    raise KeyError(f"Columns not found among: {candidates}")

def _get_feat_cols(df):
    """
    각 시점 t에 대해 내부 표준 순서로 컬럼을 반환:
    [r_top, dr_top, r_bottom, dr_bottom]
    - CSV가 r*_bot 또는 r*_bottom 어느 쪽이든 지원
    - dr*_bot / dr*_bottom도 지원
    """
    feat_cols = []
    for t in range(SEQ_T):
        r_top = _pick_col(df,  [f"r{t}_top",  f"r{t}_Top",  f"r{t}_TOP"])
        dr_top = _pick_col(df, [f"dr{t}_top", f"dr{t}_Top", f"dr{t}_TOP"])
        r_bot = _pick_col(df,  [f"r{t}_bottom",  f"r{t}_bot",  f"r{t}_Bottom",  f"r{t}_Bot",  f"r{t}_BOT"])
        dr_bot = _pick_col(df, [f"dr{t}_bottom", f"dr{t}_bot", f"dr{t}_Bottom", f"dr{t}_Bot", f"dr{t}_BOT"])
        feat_cols.extend([r_top, dr_top, r_bot, dr_bot])
    return feat_cols

class IntentDataset(Dataset):
    def __init__(self, df, mean=None, std=None, fit_scaler=False,
                 # ===== NEW: augmentation options =====
                 augment=False, aug_sigma=0.07, aug_shift_prob=0.2,
                 aug_drift_sigma=0.02, aug_mask_prob=0.1, seed=42):
        """
        df columns(예시):
          session_id, t_end, label,
          r0_bot, r0_top, dr0_bot, dr0_top, ... r4_bot, r4_top, dr4_bot, dr4_top
        또는
          r0_bottom, r0_top, dr0_bottom, dr0_top, ...
        """
        self.df = df.reset_index(drop=True)

        # 내부 표준 순서 컬럼 생성
        self.feat_cols = _get_feat_cols(self.df)

        # X: [N, T, 4]
        X = self.df[self.feat_cols].values.astype(np.float32)
        X = X.reshape(len(self.df), SEQ_T, 4)

        # label(optional)
        self.y = self.df["label"].values.astype(np.float32) if "label" in self.df.columns else None

        # 채널별 표준화 (r_top, dr_top, r_bottom, dr_bottom 각각)
        if fit_scaler:
            mean = X.reshape(-1, 4).mean(axis=0)           # [4]
            std  = X.reshape(-1, 4).std(axis=0) + 1e-8     # [4]
        self.mean = mean
        self.std  = std
        X = (X - self.mean) / self.std

        self.X = torch.from_numpy(X)  # [N, T, 4]

        # ===== NEW: augmentation params =====
        self.augment = augment
        self.aug_sigma = float(aug_sigma)            # 가우시안 노이즈 표준편차(표준화 공간)
        self.aug_shift_prob = float(aug_shift_prob)  # 좌/우 한 스텝 시프팅 확률
        self.aug_drift_sigma = float(aug_drift_sigma)# 서서히 변하는 드리프트 강도
        self.aug_mask_prob = float(aug_mask_prob)    # 한 타임스텝 마스킹 확률(=평균치로)
        self.rng = np.random.RandomState(seed)

    def __len__(self): return len(self.df)

    def __getitem__(self, idx):
        x = self.X[idx].clone()                    # [T,4]
        if self.y is None:
            return x, None
        y = torch.tensor(self.y[idx])              # scalar

        # ===== NEW: on-the-fly augmentation (train only) =====
        if self.augment:
            # (1) 미세 시간 이동: -1 또는 +1 스텝
            if self.rng.rand() < self.aug_shift_prob:
                shift = int(self.rng.choice([-1, 1]))
                x = torch.roll(x, shifts=shift, dims=0)

            # (2) 가우시안 노이즈 (표준화 공간이라 0.05~0.10 권장)
            if self.aug_sigma > 0:
                x = x + torch.randn_like(x) * self.aug_sigma

            # (3) 선형 드리프트 (느린 베이스라인 변화 모사)
            if self.aug_drift_sigma > 0:
                T = x.shape[0]
                t = torch.linspace(-1, 1, steps=T).view(T, 1)  # [-1..1]
                drift = torch.randn(1, x.shape[1]) * self.aug_drift_sigma
                x = x + t * drift

            # (4) 타임 마스크: 임의 1시점 정보를 평균(0)으로
            if self.rng.rand() < self.aug_mask_prob:
                t_idx = int(self.rng.randint(0, x.shape[0]))
                x[t_idx] = 0.0  # 표준화 후 평균=0

        return x, y


# -----------------------------
# Model
# -----------------------------
class MinimalGRU(nn.Module):
    def __init__(self, input_dim=4, hidden=64, num_layers=1, dropout=0.0):
        super().__init__()
        self.gru = nn.GRU(
            input_dim, hidden,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.0 if num_layers==1 else dropout
        )
        self.head = nn.Sequential(
            nn.Linear(hidden, 64), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(64, 1)  # 로지츠 출력 (Sigmoid는 Loss에서)
        )

    def forward(self, x):  # x: [B, T, 4]
        _, h = self.gru(x)       # h: [num_layers, B, hidden]
        h = h[-1]                # [B, hidden]
        logits = self.head(h).squeeze(-1)  # [B]
        return logits

# -----------------------------
# Train / Eval (acc만 계산)
# -----------------------------
def train_one_epoch(model, loader, optimizer, device, pos_weight=None):
    model.train()
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(device)) if pos_weight is not None else nn.BCEWithLogitsLoss()
    losses = []
    for X, y in loader:
        if y is None:
            continue
        X = X.to(device); y = y.to(device)
        optimizer.zero_grad()
        logits = model(X)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    return float(np.mean(losses)) if losses else 0.0

@torch.no_grad()
def eval_one_epoch(model, loader, device, thresh=0.5):
    """반환: (mean_loss, accuracy)"""
    model.eval()
    criterion = nn.BCEWithLogitsLoss()
    losses, correct, total = [], 0, 0
    for X, y in loader:
        if y is None:
            continue
        X, y = X.to(device), y.to(device)
        logits = model(X)
        loss = criterion(logits, y)
        losses.append(loss.item())
        probs = torch.sigmoid(logits)
        y_pred = (probs >= thresh).float()
        correct += (y_pred == y).sum().item()
        total   += y.numel()
    mean_loss = float(np.mean(losses)) if losses else 0.0
    acc = (correct / total) if total > 0 else 0.0
    return mean_loss, acc

# -----------------------------
# End-to-end training entry
# -----------------------------
def run_train(csv_path, batch_size=128, epochs=20, lr=1e-3,
              valid_ratio=0.2, seed=42, thresh=0.5, device=None):
    set_seed(seed)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    df = pd.read_csv(csv_path)

    # === 세션 단위/행 단위 분할 ===
    df_train, df_valid = group_train_valid_split(
        df, group_col="session_id", valid_ratio=valid_ratio, seed=seed
    )

    # === 스케일 파라미터 피팅 ===
    ds_tmp = IntentDataset(df_train, fit_scaler=True)
    mean, std = ds_tmp.mean, ds_tmp.std

    # === Dataset (train만 증강 적용: ENABLE_AUG) ===
    ds_train = IntentDataset(
        df_train, mean=mean, std=std, fit_scaler=False,
        augment=ENABLE_AUG,            # 증강 on/off
        aug_sigma=0.07,                # 가우시안 노이즈
        aug_shift_prob=0.2,            # ±1 step 시프트 확률
        aug_drift_sigma=0.02,          # 선형 드리프트 강도
        aug_mask_prob=0.10,            # 1시점 마스킹 확률
        seed=seed
    )
    ds_valid = IntentDataset(
        df_valid, mean=mean, std=std, fit_scaler=False, augment=False
    )

    # === DataLoader (검증은 항상 분포 그대로) ===
    dl_valid = DataLoader(ds_valid, batch_size=batch_size, shuffle=False)

    # 기본 로더 (샘플러 미사용) + 학습셋 평가용 로더
    dl_train = DataLoader(ds_train, batch_size=batch_size, shuffle=True)
    dl_train_eval = DataLoader(ds_train, batch_size=batch_size, shuffle=False)

    # === 업샘플러(ENABLE_SAMPLER) 설정 ===
    use_sampler = False
    if ENABLE_SAMPLER and "label" in df_train.columns:
        y_np = df_train["label"].to_numpy().astype(np.float32)
        pos = float(y_np.sum())
        neg = float(len(y_np) - y_np.sum())
        if pos > 0 and neg > 0:
            w_pos = neg / (pos + 1e-9)  # 라벨=1에 더 큰 가중
            weights = np.where(y_np == 1.0, w_pos, 1.0).astype(np.float32)
            sampler = WeightedRandomSampler(
                weights=weights.tolist(),
                num_samples=len(weights),           # 한 에폭 길이 = 원본 train 샘플 수
                replacement=True,
                generator=torch.Generator().manual_seed(seed)
            )
            dl_train = DataLoader(ds_train, batch_size=batch_size, sampler=sampler)
            dl_train_eval = DataLoader(ds_train, batch_size=batch_size, shuffle=False)
            use_sampler = True

    # === pos_weight: 샘플러 사용 시 과보상 방지 위해 끄는 걸 권장 ===
    if "label" in df_train.columns:
        pos_cnt = float(df_train["label"].sum())
        neg_cnt = float(len(df_train) - df_train["label"].sum())
        computed_pos_weight = torch.tensor(neg_cnt / (pos_cnt + 1e-9), dtype=torch.float32) if pos_cnt > 0 else None
    else:
        computed_pos_weight = None
    pos_weight_to_use = None if use_sampler else computed_pos_weight

    # === 모델/옵티마이저 ===
    model = MinimalGRU(input_dim=4, hidden=64, num_layers=1, dropout=0.0).to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=lr)

    # === 학습 루프 ===
    best_acc, best_state = -1.0, None
    print(f"Start training!  train={len(df_train)}  valid={len(df_valid)}  "
          f"(pos={int(df_train['label'].sum()) if 'label' in df_train.columns else 'NA'})")

    for ep in range(1, epochs + 1):
        # ---- Train(step) ----
        tr_loss = train_one_epoch(model, dl_train, optim, device, pos_weight=pos_weight_to_use)
        # 샘플러/증강 없는 로더로 공정하게 Train 지표 산출
        tr_loss_eval, tr_acc = eval_one_epoch(model, dl_train_eval, device, thresh=thresh)
        # ---- Valid ----
        va_loss, va_acc = eval_one_epoch(model, dl_valid, device, thresh=thresh)

        print(
            f"[Epoch {ep:02d}] "
            f"Train: loss={tr_loss:.4f} (eval_loss={tr_loss_eval:.4f}) acc={tr_acc:.3f}  |  "
            f"Valid: loss={va_loss:.4f} acc={va_acc:.3f}",
            flush=True
        )

        # 베스트(검증 정확도) 갱신 시 체크포인트 보관
        if va_acc > best_acc:
            best_acc = va_acc
            best_state = {
                "model": model.state_dict(),
                "mean": mean.tolist(),
                "std": std.tolist(),
                "thresh": thresh
            }

    # === 저장 ===
    os.makedirs("artifacts", exist_ok=True)
    if best_state is None:
        best_state = {
            "model": model.state_dict(),
            "mean": mean.tolist(),
            "std": std.tolist(),
            "thresh": thresh
        }
    torch.save(best_state["model"], "artifacts/intent_gru.pt")
    with open("artifacts/scale.json", "w") as f:
        json.dump({"mean": best_state["mean"], "std": best_state["std"],
                   "thresh": best_state["thresh"]}, f)
    print("Saved model to artifacts/intent_gru.pt and artifacts/scale.json")

    return model, (mean, std), thresh


# -----------------------------
# Inference (single window)
# -----------------------------
@torch.no_grad()
def predict_single_window(seq_5x4, model_path="artifacts/intent_gru.pt",
                          scale_path="artifacts/scale.json", device=None):
    """
    seq_5x4: shape (5,4) = [[r0_top,dr0_top,r0_bottom,dr0_bottom], ... [r4_top,dr4_top,r4_bottom,dr4_bottom]]
    return: p_intent (0~1)
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    state_dict = torch.load(model_path, map_location=device)
    with open(scale_path, "r") as f: sc = json.load(f)
    mean = torch.tensor(sc["mean"], dtype=torch.float32)   # [4]
    std  = torch.tensor(sc["std"],  dtype=torch.float32)   # [4]

    model = MinimalGRU(input_dim=4, hidden=64, num_layers=1, dropout=0.0).to(device)
    model.load_state_dict(state_dict)
    model.eval()

    x = torch.tensor(seq_5x4, dtype=torch.float32).view(1, 5, 4)  # [1,5,4]
    x = (x - mean) / std
    logits = model(x)
    prob = torch.sigmoid(logits).item()
    return prob

# -----------------------------
# PyCharm에서 그냥 실행
# -----------------------------
if __name__ == "__main__":
    # 여기 설정값(CSV_PATH 등)만 바꾸고
    model, (mean, std), _ = run_train(
        csv_path=CSV_PATH,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        lr=LR,
        valid_ratio=VALID_RATIO,
        seed=SEED,
        thresh=THRESH
    )


