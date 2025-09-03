# modules/intent_runtime.py
import json, math
import numpy as np
import torch
from torch import nn
from dataclasses import dataclass
from collections import deque

# ===== 학습 때 쓴 모델과 동일 =====
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
            nn.Linear(64, 1)
        )
    def forward(self, x):  # x: [B,T,4]
        _, h = self.gru(x)
        h = h[-1]
        return self.head(h).squeeze(-1)

@dataclass
class RuntimeCfg: # 추론 시 세팅
    window_sec: int = 5
    downsample_hz: float = 1.0
    thresh: float = 0.5
    coords_unit: str = "m"          # 들어오는 (x,y) 단위: "m" or "cm"
    use_cuda_if_available: bool = True

class DoorIntentRuntime:
    """
    log(ts, x, y)를 반복 호출하면 윈도우가 찰 때마다 추론 결과를 반환.
    앵커 좌표는 meter 단위로 전달해줘.
    """
    def __init__(self, anchor1_xy_m, anchor2_xy_m,
                 model_path="../artifacts/intent_gru.pt",
                 scale_path="../artifacts/scale.json",
                 cfg: RuntimeCfg = RuntimeCfg()):
        self.cfg = cfg
        self._xy_scale = 0.01 if cfg.coords_unit.lower() == "cm" else 1.0
        self.a1x, self.a1y = float(anchor1_xy_m[0]), float(anchor1_xy_m[1])
        self.a2x, self.a2y = float(anchor2_xy_m[0]), float(anchor2_xy_m[1])
        self.hist = deque(maxlen=self.cfg.window_sec)
        self.prev_raw = None
        self._last_emit_ts = None
        self._emit_interval = 1.0 / max(1e-6, self.cfg.downsample_hz)
        self.on_prediction = None

        self.device = "cuda" if (self.cfg.use_cuda_if_available and torch.cuda.is_available()) else "cpu"
        # scale.json
        with open(scale_path, "r") as f: # thresh 있으면 우선 적용
            sc = json.load(f)
        self.mean = torch.tensor(sc["mean"], dtype=torch.float32).to(self.device)
        self.std  = torch.tensor(sc["std"],  dtype=torch.float32).to(self.device)
        self.thresh = float(sc.get("thresh", self.cfg.thresh))
        # model
        self.model = MinimalGRU(input_dim=4, hidden=64, num_layers=1, dropout=0.0).to(self.device)
        state = torch.load(model_path, map_location="cpu")
        self.model.load_state_dict(state)
        self.model.eval()

    def set_callback(self, fn): self.on_prediction = fn
    def reset(self):
        self.hist.clear(); self.prev_raw=None; self._last_emit_ts=None

    def log(self, ts: float, x, y):
        # 좌표 단위 보정
        x_m = float(x) * self._xy_scale
        y_m = float(y) * self._xy_scale

        r_top = self._euclid(x_m, y_m, self.a1x, self.a1y)
        r_bot = self._euclid(x_m, y_m, self.a2x, self.a2y)

        if self.prev_raw is None:
            dr_top = 0.0; dr_bot = 0.0
        else:
            prev_ts, _, _, prev_r_top, prev_r_bot = self.prev_raw
            dt = max(1e-6, ts - prev_ts)
            dr_top = (r_top - prev_r_top) / dt
            dr_bot = (r_bot - prev_r_bot) / dt

        # 1 Hz로만 밀어넣기
        if (self._last_emit_ts is None) or (ts - self._last_emit_ts >= self._emit_interval):
            self.hist.append({"ts": ts, "r_top": r_top, "dr_top": dr_top, "r_bot": r_bot, "dr_bot": dr_bot})
            self._last_emit_ts = ts
            if len(self.hist) == self.cfg.window_sec:
                seq = [[w["r_top"], w["dr_top"], w["r_bot"], w["dr_bot"]] for w in self.hist]
                prob = self._infer_seq(np.asarray(seq, np.float32))
                res = {"t_end": float(self.hist[-1]["ts"]), "prob": float(prob),
                       "over_thresh": bool(prob >= self.thresh), "seq_5x4": seq}
                if self.on_prediction:
                    try: self.on_prediction(res)
                    except Exception as e: print(f"[WARN] on_prediction error: {e}")
                self.prev_raw = (ts, x_m, y_m, r_top, r_bot)
                return res

        self.prev_raw = (ts, x_m, y_m, r_top, r_bot)
        return None

    @torch.no_grad()
    def _infer_seq(self, seq_5x4: np.ndarray) -> float:
        x = torch.from_numpy(seq_5x4).unsqueeze(0).to(self.device)      # [1,5,4]
        x = (x - self.mean) / (self.std + 1e-8)
        logits = self.model(x)
        prob = torch.sigmoid(logits).item()
        print(f"[DEBUG] infer prob={prob:.4f}")  # ← 여기 추가
        return torch.sigmoid(logits).item()

    @staticmethod
    def _euclid(x1,y1,x2,y2): return math.hypot(x1-x2, y1-y2)
