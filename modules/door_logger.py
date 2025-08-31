import csv
from datetime import datetime

from shapely import LineString
from shapely.geometry import Point, Polygon
import pandas as pd
from collections import deque
import re
from dataclasses import dataclass
from datetime import datetime


@dataclass
class DSConfig:
    window_sec: int = 5  # 윈도우 길이(초)
    delta_sec: float = 2.0  # crossing 이후 라벨=1 지속시간
    stride_sec: int = 1  # (옵션) 샘플 stride
    hz: float = 1.0  # 샘플링 주파수(레거시)


class DoorIntentLogger:
    def __init__(self, vehicle_polygon, anchor1_xy, anchor2_xy,
                 session_id="S1", cfg=DSConfig()):
        self.prev_x = None
        self.prev_y = None
        self.cfg = cfg
        self.session_id = session_id

        # 차량 폴리곤 (cm → m 변환)
        vehicle_polygon_m = [(x / 100.0, y / 100.0) for (x, y) in vehicle_polygon]
        self.vehicle_poly = Polygon(vehicle_polygon_m)

        # 문 위치 (m 단위로 전달된다고 가정)
        self.anchor1 = Point(anchor1_xy)  # 윗문
        self.anchor2 = Point(anchor2_xy)  # 아랫문

        # 문 반경 (m)
        self.door_r = 2
        self.top_door_area = self.anchor1.buffer(self.door_r)
        self.bot_door_area = self.anchor2.buffer(self.door_r)

        # 기록용
        self.hist = deque()  # 최근 window_sec 샘플
        self.prev_inside = None  # 이전 차량 내부 여부
        self.prev_ts = None
        self.crossings = []  # crossing 발생 시각 기록
        self.samples = []  # 최종 샘플
        self.last_emit_ts = None  # 다운샘플링 타이머

        # --- CSV 파일 자동 생성 ---
        nowtime = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.csv_path = f"intent_door_{nowtime}.csv"
        self.csv_file = open(self.csv_path, "w", newline="", encoding="utf-8")
        self.csv_writer = None  # 첫 샘플 올 때 초기화

    def log(self, ts, x_cm, y_cm, inside_vehicle=None):
        """
        실시간 로그 처리
        - ts: 초 단위 timestamp
        - x_cm, y_cm: 위치 (cm 단위)
        """

        p = Point(x_cm, y_cm)

        # 앵커(문)까지 거리
        r_top = p.distance(self.anchor1)
        r_bot = p.distance(self.anchor2)

        # dr 계산
        if self.prev_ts is not None and self.hist:
            dt = max(1e-6, ts - self.prev_ts)
            prev = self.hist[-1]
            dr_top = (r_top - prev["r_top"]) / dt
            dr_bot = (r_bot - prev["r_bot"]) / dt
        else:
            dr_top = dr_bot = 0.0

        # 차량 폴리곤 안/밖
        if inside_vehicle is not None:
            inside_vehicle_now = inside_vehicle
        else:
            inside_vehicle_now = self.vehicle_poly.covers(p)

        # --- crossing 판정 ---
        crossing_now = False
        if self.prev_inside is not None and inside_vehicle_now != self.prev_inside:
            prev_p = Point(self.prev_x, self.prev_y)
            path = LineString([prev_p, p])

            # (1) path가 원을 스쳤거나
            intersects_door = path.intersects(self.top_door_area) or path.intersects(self.bot_door_area)

            # (2) prev와 now 둘 다 문 반경 안이거나
            prev_in_door = (self.anchor1.distance(prev_p) <= self.door_r or
                            self.anchor2.distance(prev_p) <= self.door_r)
            now_in_door = (r_top <= self.door_r or r_bot <= self.door_r)

            if intersects_door or (prev_in_door or now_in_door):
                self.crossings.append(ts)

        # 다운샘플링 (1Hz)
        if self.last_emit_ts is None or ts - self.last_emit_ts >= 1.0:
            self.hist.append({
                "ts": ts,
                "r_top": r_top, "dr_top": dr_top,
                "r_bot": r_bot, "dr_bot": dr_bot,
            })
            self.last_emit_ts = ts

            # 오래된 샘플 제거
            while len(self.hist) > self.cfg.window_sec:
                self.hist.popleft()

            # --- 학습 샘플 생성 ---
            if len(self.hist) >= self.cfg.window_sec:
                window = list(self.hist)[-self.cfg.window_sec:]
                t_end = window[-1]["ts"]

                # 입력 시퀀스 펼치기
                seq_flat = {}
                for i, w in enumerate(window):
                    seq_flat[f"r{i}_bot"] = w["r_bot"]
                    seq_flat[f"r{i}_top"] = w["r_top"]
                    seq_flat[f"dr{i}_bot"] = w["dr_bot"]
                    seq_flat[f"dr{i}_top"] = w["dr_top"]

                # --- 라벨 판정 ---
                label = 0
                if self.crossings:
                    last_cross = self.crossings[-1]
                    # crossing이 t_end 이후 delta_sec 이내에 발생하면 label=1
                    if t_end - self.cfg.delta_sec <= last_cross <= t_end:
                        label = 1
                        print(f"[LABEL=1] t_end={t_end:.2f}, last_cross={last_cross:.2f}, ")
                    else:
                        print(f"[LABEL=0] t_end={t_end:.2f}, last_cross={last_cross:.2f}, ")

                sample = {
                    "session_id": self.session_id,
                    "t_end": t_end,
                    "label": label,
                    **seq_flat
                }
                self.samples.append(sample)

                # --- CSV 실시간 기록 ---
                if self.csv_writer is None:
                    # 첫 번째 샘플일 때 헤더 작성
                    fieldnames = list(sample.keys())
                    self.csv_writer = csv.DictWriter(self.csv_file, fieldnames=fieldnames)
                    self.csv_writer.writeheader()
                self.csv_writer.writerow(sample)
                self.csv_file.flush()

        # 상태 갱신
        self.prev_ts = ts
        self.prev_inside = inside_vehicle_now
        self.prev_x, self.prev_y = x_cm, y_cm

    def export_csv(self, path_csv="intent_door.csv", keep_session=True):
        """누적된 샘플을 CSV로 저장"""
        if not self.samples:
            print("[INFO] 수집 샘플 없음")
            return
        df = pd.DataFrame(self.samples)

        if not keep_session and "session_id" in df.columns:
            df = df.drop(columns=["session_id"])

        # 열 순서 정렬
        meta = [c for c in ["session_id", "t_end", "label"] if c in df.columns]

        def _key(col):
            m = re.match(r"^(r|dr)(\d+)_(top|bot)$", col)
            if not m:
                return (10 ** 9, 9, col)
            kind, idx, side = m.groups()
            return (int(idx), 0 if kind == "r" else 1, side)

        seq_cols = sorted([c for c in df.columns if c not in meta], key=_key)
        df = df[meta + seq_cols]

        df.to_csv(path_csv, index=False)
        print(f"[OK] CSV 저장: {path_csv}")

    def close(self):
        """ 프로그램 종료 시 파일 닫기 """
        if self.csv_file:
            self.csv_file.close()
            print(f"[OK] CSV 파일 닫음: {self.csv_path}")