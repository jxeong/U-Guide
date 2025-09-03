# ///////////////////////////////////////////////////////////////
import csv
import ctypes
import math
import platform
import shutil
import sys
from PySide6.QtGui import QPainterPath


from PySide6.QtGui import QPixmap, QColor, QPainter, Qt
from PySide6.QtWidgets import QMessageBox, QDialog, QListWidget, QPushButton, QVBoxLayout
from PySide6.QtCore import QUrl, QTimer, QMetaObject
# MAIN FILE
from main import *
from modules import Settings
from PySide6.QtWidgets import QLabel, QFileDialog, QMessageBox, QToolTip
import json
import serial
from serial.tools import list_ports
from modules.utils import resource_path
from modules.uwb_functions import Calculation
from modules.serial_handler import DualSerialHandler
from datetime import datetime
import re
import sqlite3
import os
import time
import threading
from shapely.geometry import Point, Polygon, LineString
import glob
from modules.intent_runtime import DoorIntentRuntime, RuntimeCfg
import pyttsx3
from PySide6.QtGui import QPixmap, QColor, QPainter, Qt, QPen, QFont, QImage, QTransform
from PySide6.QtCore import QUrl, QTimer, QMetaObject, QRectF, QPointF
from PySide6.QtGui import QPolygonF



def get_existing_db_path():
    """ 실행 폴더 또는 모듈 폴더 내 존재하는 .db 파일 중 첫 번째 경로 반환 (복사 안 함) """
    if getattr(sys, 'frozen', False):
        base_path = os.path.dirname(sys.executable)
    else:
        base_path = os.path.dirname(__file__)

    # 현재 경로에서 .db 파일 탐색
    db_files = [f for f in os.listdir(base_path) if f.endswith(".db")]

    if not db_files:
        raise FileNotFoundError("실행 폴더에 .db 파일이 존재하지 않습니다.")

    # 첫 번째 db 파일 경로 반환
    return os.path.join(base_path, db_files[0])


db_path = get_existing_db_path()


def get_drive_list_full():
    DRIVE_TYPE = {
        0: "알 수 없음",
        1: "루트 디렉터리 없음",
        2: "이동식 디스크",  # USB 등
        3: "로컬 디스크",  # HDD, SSD
        4: "네트워크 드라이브",  # NAS, 공유폴더
        5: "CD-ROM",
        6: "RAM 디스크"
    }

    drives = []
    for part in psutil.disk_partitions():
        path = part.device  # 'C:\\'
        name_buf = ctypes.create_unicode_buffer(1024)
        fs_buf = ctypes.create_unicode_buffer(1024)
        drive_type_code = ctypes.windll.kernel32.GetDriveTypeW(ctypes.c_wchar_p(path))
        type_name = DRIVE_TYPE.get(drive_type_code, "Unknown")

        try:
            ctypes.windll.kernel32.GetVolumeInformationW(
                ctypes.c_wchar_p(path),
                name_buf,
                ctypes.sizeof(name_buf),
                None, None, None,
                fs_buf,
                ctypes.sizeof(fs_buf)
            )
            label = name_buf.value
        except Exception:
            label = ""

        drive_letter = path.strip("\\")  # 'C:'
        if label:
            display_name = f"{label} ({drive_letter}) - {type_name}"
        else:
            display_name = f"{drive_letter} - {type_name}"

        drives.append((display_name, path))
    return drives


def show_drive_select_dialog(parent=None):
    dialog = QDialog(parent)
    dialog.setWindowTitle("드라이브 선택")
    dialog.setMinimumWidth(300)

    layout = QVBoxLayout(dialog)

    list_widget = QListWidget()
    drives = get_drive_list_full()
    for label, path in drives:
        list_widget.addItem(label)
    layout.addWidget(list_widget)

    ok_button = QPushButton("선택")
    layout.addWidget(ok_button)

    selected_drive = None  # 기본 None

    def on_ok():
        nonlocal selected_drive
        selected_item = list_widget.currentItem()
        if selected_item:
            text = selected_item.text()

            # 괄호 안 드라이브 문자 있는 경우: (D:)
            if '(' in text and ')' in text:
                drive_letter = text[text.find('(') + 1:text.find(')')]
            else:
                # 괄호 없으면 맨 앞 단어 사용: C:, E: 등
                drive_letter = text.split()[0]

            selected_drive = drive_letter + '\\'
            dialog.accept()
        else:
            QMessageBox.warning(dialog, "경고", "드라이브를 선택해주세요.")

    ok_button.clicked.connect(on_ok)

    result = dialog.exec()

    # 선택된 경우에만 반환, X 누르면 None 유지
    if result == QDialog.Accepted and selected_drive:
        return selected_drive
    return None


class AppFunctions:
    def __init__(self, parent):
        self.parent = parent
        self.ui = parent.ui

        # 변수 초기화
        self.dual_serial_handler = None
        self.detected_tags = None
        self.current_workspace_name = None
        self.scale_ratio = None
        self.tag_position = None
        self.calculation = None  # Calculation 클래스 초기화 지연
        self.preview_vertex_position = None  # vertex 이동 중 미리보기 좌표

        self.anchor_positions = []
        self.anchor_labels = {}
        self.tag_positions = {}
        self.tag_connection_status = {}
        self.tag_names = {}  # 태그 이름 저장
        self.tag_status_list = []  # 태그 상태 저장
        self.anchor_data = {}
        self.vertex_data = {}
        self.prev_positions = {}  # 속도 제한 변수 1
        self.prev_timestamps = {}  # 속도 제한 변수 2

        # 데이터베이스 초기화
        self.start_serial_connection()
        self.tags_in_danger_zone = set()
        self.db_path = get_existing_db_path()
        self.initialize_database()
        self.initialize_anchor_labels()

        self.workspace_loaded = False  # 워크스페이스 로드 상태
        self.danger_color = QColor(255, 0, 0)  # 기본 빨간색 반투명
        self.workspace_color = QColor(130, 163, 196, 40)  # 기본 작업 공간 색상

        # 태그 관련 초기화
        self.parent.ui.g_anchorNum.valueChanged.connect(self.update_anchor_count)
        self.parent.ui.pushButton_4.pressed.connect(self.save_as_new_workspace)
        self.parent.ui.pushButton_3.pressed.connect(self.edit_workspace)

        self.parent.ui.workspace.paintEvent = self.paint_workspace

        self.parent.ui.i_anchorSelect.currentIndexChanged.connect(self.update_anchor_position)
        self.parent.ui.j_anchorX.valueChanged.connect(self.save_anchor_position)
        self.parent.ui.k_anchorY.valueChanged.connect(self.save_anchor_position)
        self.parent.ui.vertexSelect.currentIndexChanged.connect(self.update_vertex_position)
        self.parent.ui.vertexX.valueChanged.connect(self.save_vertex_position)
        self.parent.ui.vertexY.valueChanged.connect(self.save_vertex_position)
        self.parent.ui.g_anchorNum.valueChanged.connect(self.update_visible_anchors)
        self.parent.ui.pushButton.pressed.connect(self.open_existing_workspace)

        # dataset  수집
#        self.parent.ui.btnExportCsv.clicked.connect(lambda: self.export_intent_dataset_csv())

        # inactive, active Button 관련
        self.tag_in_danger_zone = False

        # anchor, vertex 이동 + tooltip
        self.parent.ui.workspace.mousePressEvent = self.handle_workspace_click
        self.parent.ui.workspace.setMouseTracking(True)
        self.parent.ui.workspace.mouseMoveEvent = self.show_tooltip_during_move

        QTimer.singleShot(0, self.redraw_workspace_after_init)

        # 태그 타임아웃 체크 타이머 (태그가 끊겼을 때도 감지)
        self.tag_timeout_timer = QTimer(self.parent)
        self.tag_timeout_timer.timeout.connect(self.check_tag_timeouts)
        self.tag_timeout_timer.start(1000)  # 1초마다 검사

        self.initialize_kalman_filters()

        #승하차 추론
        self.last_alert_time =0

        self.bg_image = None  # QImage
        self.bg_opacity = 0.9  # 0.0~1.0
        self.bg_image_path = None  # 선택(로그/재로딩용)

    def set_background_png(self, path: str, opacity: float = 0.9):
        """외부에서 호출: PNG 경로만 주면 vertex 폴리곤에 맞춰 그림."""
        img = QImage(path)
        if img.isNull():
            QMessageBox.warning(self.parent, "오류", f"이미지 로드 실패: {path}")
            return False
        self.bg_image = img
        self.bg_opacity = max(0.0, min(1.0, opacity))
        self.bg_image_path = path
        self.parent.ui.workspace.update()
        return True

    def clear_background_png(self):
        self.bg_image = None
        self.bg_image_path = None
        self.parent.ui.workspace.update()

    def _draw_png_in_vertices(self, painter: QPainter):
        if self.bg_image is None:
            return
        if not hasattr(self, "vertex_points") or not self.vertex_points:
            return

        pts = list(self.vertex_points)
        if hasattr(self, "moving_vertex") and self.moving_vertex and self.preview_vertex_position:
            try:
                idx = int(self.moving_vertex.split(" ")[1]) - 1
                pts[idx] = self.preview_vertex_position
            except Exception:
                pass

        qpts = [QPointF(x, y) for (x, y) in pts]
        poly = QPolygonF(qpts)
        bbox: QRectF = poly.boundingRect()
        if bbox.width() <= 0 or bbox.height() <= 0:
            return

        # === 여기서는 원본 비율 무시 ===
        target_w = bbox.width()
        target_h = bbox.height()
        x, y = bbox.x(), bbox.y()

        painter.save()
        clip_path = QPainterPath()
        clip_path.addPolygon(poly)  # 다각형 내부만 보이게 clip
        painter.setClipPath(clip_path)
        painter.setOpacity(self.bg_opacity)
        # 이미지를 bbox 크기에 맞춰 강제 리사이즈
        painter.drawImage(QRectF(x, y, target_w, target_h), self.bg_image)
        painter.restore()

    def load_background_image(self):
        path, _ = QFileDialog.getOpenFileName(self.parent, "PNG/JPG 선택", "", "Images (*.png *.jpg *.jpeg)")
        if not path:
            return
        img = QImage(path)
        if img.isNull():
            QMessageBox.warning(self.parent, "오류", "이미지 로드 실패")
            return
        self.bg_image = img
        self.bg_opacity = 0.85  # 필요시 투명도 조절
        self.parent.ui.workspace.update()

    def clear_background_image(self):
        self.bg_image = None
        self.parent.ui.workspace.update()

    def play_tts(self, text, cooldown=5):
        """TTS 알림 (스레드 분리 + 쿨다운 적용)"""
        now = time.time()
        if now - self.last_alert_time < cooldown:
            return

        def tts_worker():
            try:
                engine = pyttsx3.init()
                engine.say(text)
                engine.runAndWait()
            except Exception as e:
                print(f"[ERROR] TTS 실패: {e}")

        # 스레드에서 실행 → 메인 루프(tag 업데이트)와 동시에 동작
        threading.Thread(target=tts_worker, daemon=True).start()
        self.last_alert_time = now

    def _extract_top_bottom_edges_from_vertices(self):
        """
        self.vertex_points (화면 좌표, y 아래로 증가)에서
        '윗변(Top edge)'과 '아랫변(Bottom edge)'의 두 꼭짓점을 추출한다.
        - 윗변: y가 가장 작은 두 점
        - 아랫변: y가 가장 큰 두 점
        반환: (UL, UR, LL, LR) 튜플 (각 원소는 (x, y))
        """
        if not hasattr(self, "vertex_points") or len(self.vertex_points) < 4:
            return None

        pts = list(self.vertex_points)  # [(x,y), ...]
        # y 오름차순(위쪽이 먼저)
        pts_sorted_by_y = sorted(pts, key=lambda p: p[1])
        top2 = sorted(pts_sorted_by_y[:2], key=lambda p: p[0])  # x로 정렬 → (UL, UR)
        bot2 = sorted(pts_sorted_by_y[-2:], key=lambda p: p[0])  # x로 정렬 → (LL, LR)

        UL, UR = top2[0], top2[1]
        LL, LR = bot2[0], bot2[1]
        return UL, UR, LL, LR


    def _build_or_update_intent_runtime(self):
        """
        DoorIntentRuntime: 실시간 1Hz 윈도우 → GRU 추론
        - anchor1=윗문, anchor2=아랫문 (meter 단위)
        """
        if not hasattr(self, "anchor_positions") or len(self.anchor_positions) < 3:
            return
        try:
            anchor1_xy_m = self.anchor_positions[1]  # 윗문
            anchor2_xy_m = self.anchor_positions[2]  # 아랫문
        except Exception:
            return

        # 좌표 단위: calculate_position_for_tag()가 반환하는 x,y가 'm'이면 coords_unit="m"
        cfg = RuntimeCfg(window_sec=5, downsample_hz=1.0, coords_unit="m")
        self.intent_runtime = DoorIntentRuntime(
            anchor1_xy_m=anchor1_xy_m,
            anchor2_xy_m=anchor2_xy_m,
            model_path="artifacts/intent_gru.pt",
            scale_path="artifacts/scale.json",
            cfg=cfg
        )


        # 추론 콜백: UI 갱신/비프/로그 등
        def _on_pred(res):
            try:
                if hasattr(self.parent.ui, "intentProb"):
                    self.parent.ui.intentProb.setText(f"{res['prob']:.2f}")
                if res["over_thresh"]:
                    # 이제 블로킹 안 됨 → tag_position 계속 업데이트됨
                    self.play_tts(
                        "교통약자 승객의 승하차를 위해 공간을 양보해 주시면 감사하겠습니다.",
                        cooldown=5
                    )
            except Exception as e:
                print(f"[WARN] intent on_pred UI update failed: {e}")

        self.intent_runtime.set_callback(_on_pred)  # ← 여기에 있음

    # ///////////////////////////////////////////////////////////////
    # 시리얼통신 설정
    # ///////////////////////////////////////////////////////////////

    # 시리얼 포트 연결 함수
    def start_serial_connection(self):
        if self.dual_serial_handler:
            self.dual_serial_handler.disconnect_all()  # 현재 연결된 포트를 닫는다
            self.dual_serial_handler = None

        # DualSerialHandler 생성 및 연결
        self.dual_serial_handler = DualSerialHandler(
            uwb_callback=self.process_serial_data,
            parent=self
        )
        self.dual_serial_handler.connect_all()

    # 시리얼 데이터 처리 함수
    def process_serial_data(self, data):
        try:
            # print(f'raw data: {data}')
            anchor_count = self.parent.ui.g_anchorNum.value()

            # 현재 감지된 태그 목록 저장
            if not hasattr(self, "detected_tags") or self.detected_tags is None:
                self.detected_tags = {}  # {tag_id: last_seen_time}

            current_detected_tags = set()  # 이번 데이터에서 감지된 태그 목록
            current_time = time.time()  # 현재 시간 (초 단위)

            # 여러 개의 태그를 인식할 수 있도록 태그 ID별 데이터 분리
            tag_matches = re.findall(r"tid:(\d+),mask:[^,]+,seq:(\d+),range:\(([^)]+)\),rssi:\(([^)]+)\)", data)
            if not tag_matches:
                return

            for tag_match in tag_matches:
                tag_id = int(tag_match[0])  # 태그 ID
                range_values = list(map(float, tag_match[2].split(",")))[:anchor_count]  # range 값 파싱
                rssi_values = list(map(float, tag_match[3].split(",")))[:anchor_count]  # rssi 값 파싱

                current_detected_tags.add(tag_id)
                self.detected_tags[tag_id] = current_time  # 태그의 마지막 감지 시간 업데이트

                # 특정 앵커의 range와 rssi 값이 모두 0인지 확인 후 UI 업데이트
                self.update_anchor_status(range_values, rssi_values, anchor_count)

                # 태그별로 독립적인 좌표 계산 수행
                x, y = self.calculate_position_for_tag(tag_id, range_values, rssi_values, anchor_count)
                if x is not None and y is not None:
                    self.update_tag_position(x, y, tag_id)

        except (KeyError, ValueError, IndexError) as e:
            print(f"[ERROR] 데이터 처리 중 오류 발생: {e}")

    # ///////////////////////////////////////////////////////////////
    # 태그 관련
    # ///////////////////////////////////////////////////////////////

    # 3초 이상 미감지 태그 삭제 함수
    def check_tag_timeouts(self):
        """
        태그 감지 시간 기반으로 일정 시간 이상 경과한 태그를 삭제.
        """
        current_time = time.time()
        timeout = 3  # 3초

        for tag_id in list(self.detected_tags.keys()):
            last_seen = self.detected_tags[tag_id]
            if current_time - last_seen > timeout:
                self.update_tag_status(tag_id, "inactive")
                if tag_id in self.tag_positions:
                    del self.tag_positions[tag_id]
                if tag_id in self.tag_in_danger_dict:
                    del self.tag_in_danger_dict[tag_id]
                if tag_id in self.tags_in_danger_zone:
                    self.tags_in_danger_zone.remove(tag_id)
                del self.detected_tags[tag_id]

        self.update_people_count()
        self.parent.ui.workspace.update()

    # 태그 위치 계산 함수
    def calculate_position_for_tag(self, tag_id, range_values, rssi_values, anchor_count):
        try:
            MIN_VALID_RANGE = -0.2  # 10cm #필터 적용해서 뺀 값까지 생각해서 min값 정하기
            MAX_VALID_RANGE = 30.0  # 30m

            valid_anchors = []  # 위치 계산용
            filter_only_anchors = []  # 허상 제거용 (너무 가깝거나 먼 값)
            all_anchors = []  # 거리 평가용 (전체 앵커)

            for i in range(anchor_count):
                range_val = range_values[i]
                rssi_val = rssi_values[i] if i < len(rssi_values) else None

                if range_val > 0 and (rssi_val is None or rssi_val > -90):
                    anchor_pos = self.anchor_positions[i]
                    corrected_range = self.calculation.apply_correction_and_particle_single(range_val, tag_id, i)

                    if corrected_range:
                        anchor_data = {
                            "index": i,
                            "range": corrected_range,
                            "position": anchor_pos
                        }

                        all_anchors.append(anchor_data)

                        if MIN_VALID_RANGE <= corrected_range <= MAX_VALID_RANGE:
                            valid_anchors.append(anchor_data)
                        else:
                            filter_only_anchors.append(anchor_data)

            # 유효한 앵커가 3개 미만이면 위치 계산 중단
            if len(valid_anchors) < 2:
                print(f"[경고] 유효 앵커 3개 미만임 → 위치 계산 불가")
                return None, None

            # 위치 계산
            x, y = self.calculation.generalized_trilateration(
                valid_anchors=valid_anchors,
                all_anchors=all_anchors
            )

            # # 위치 계산 결과가 유효한 경우, 속도 제한 코드 추가
            if x is not None and y is not None:
                now = time.time()
                prev_pos = self.prev_positions.get(tag_id)
                prev_time = self.prev_timestamps.get(tag_id)

                if prev_pos and prev_time:
                    dist = math.sqrt((x - prev_pos[0]) ** 2 + (y - prev_pos[1]) ** 2)
                    time_diff = now - prev_time

                    speed = dist / time_diff
                    if speed > 2:  # 사람의 최대 이동 속도 m/s
                        return None, None

                # 위치 기록 갱신
                self.prev_positions[tag_id] = (x, y)
                self.prev_timestamps[tag_id] = now

            return x, y

        except (KeyError, ValueError, IndexError) as e:
            return None, None

    # 태그 위치 업데이트+화면 갱신, 위험구역 포함 여부 확인 함수
    def update_tag_position(self, x, y, tag_index):
        """
        태그 ID별 위치를 업데이트하고 화면을 갱신합니다.
        """
        if not hasattr(self, "tag_positions"):
            self.tag_positions = {}  # 태그 위치 저장 딕셔너리

        if not hasattr(self, "tag_in_danger_dict"):
            self.tag_in_danger_dict = {}  # 태그별 위험 상태 저장 {tag_id: True/False}

        # 위험 구역 진입 여부 판단
        was_in_danger = self.tag_in_danger_dict.get(tag_index, False)
        is_now_in_danger = False
        if hasattr(self, "vertex_points"):
            danger_polygon = Polygon([
                ((vx - self.workspace_box.x()) / self.scale_ratio,
                 (vy - self.workspace_box.y()) / self.scale_ratio)
                for vx, vy in self.vertex_points
            ])
            tag_point = Point(x, y)
            is_now_in_danger = danger_polygon.contains(tag_point)

        self.tag_in_danger_dict[tag_index] = is_now_in_danger

        # 작업 공간 박스의 오프셋 및 크기 가져오기
        x_offset = self.workspace_box.x()
        y_offset = self.workspace_box.y()

        # 태그 좌표 변환
        x_scaled = x * self.scale_ratio + x_offset
        y_scaled = y * self.scale_ratio + y_offset

        # 태그 위치 업데이트
        self.tag_positions[tag_index] = (x_scaled, y_scaled)

        # 현재 태그가 위험 구역 안에 있는지 확인
        is_in_danger = False  # 기본값 (위험하지 않음)
        if hasattr(self, "vertex_points"):
            danger_polygon = Polygon(self.vertex_points)  # 다각형 객체 생성
            tag_point = Point(x_scaled, y_scaled)  # 태그 위치 객체 생성
            is_in_danger = danger_polygon.contains(tag_point)  # 다각형 내부 확인

        # 개별 태그 상태 업데이트
        self.tag_in_danger_dict[tag_index] = is_in_danger

        if is_in_danger:
            self.tags_in_danger_zone.add(tag_index)
            self.update_tag_status(tag_index, "danger")
        else:
            if tag_index in self.tags_in_danger_zone:
                self.tags_in_danger_zone.remove(tag_index)
            self.update_tag_status(tag_index, "active")

        # 전체 태그 중 하나라도 danger 상태인지 확인
        self.tag_in_danger_zone = any(self.tag_in_danger_dict.values())

        # QFrame 갱신
        self.parent.ui.workspace.update()

        # ==== [데이터셋 로깅] 문 기준 r,dr 기록 + crossing 라벨용 시간 축적 ====
        try:
            if hasattr(self, "intent_runtime"):
                ts = time.time()
                res = self.intent_runtime.log(ts=ts, x=x, y=y)  # 여기서 추론
                if res is not None:
                    print(f"[DEBUG] t={res['t_end']:.2f}, prob={res['prob']:.3f}, over={res['over_thresh']}")
        except Exception as e:
            print(f"[WARN] intent_runtime log failed: {e}")

    # 위험구역 내 인원 카운트 함수
    def update_people_count(self):
        if not hasattr(self.parent.ui, "withTagPeople") or self.parent.ui.withTagPeople is None:
            return

        if not hasattr(self, "vertex_points") or not self.vertex_points:
            return

        # 위험 구역 다각형 정의
        danger_polygon = Polygon(self.vertex_points)

        # 위험 구역 내 태그들 찾기
        people_in_zone = [
            tag_id for tag_id, (x, y) in self.tag_positions.items()
            if danger_polygon.contains(Point(x, y))
        ]
        # 태그 소지자수
        num_people = len(people_in_zone)

        # UI 업데이트는 항상 수행
        self.parent.ui.withTagPeople.setText(f"{num_people}")

        if num_people > 0:
            self.parent.ui.linePeople.setStyleSheet(
                "border: 1px solid white; color: red; font-weight: bold; font-size: 16px; qproperty-alignment: AlignCenter;"
            )
        else:
            self.parent.ui.linePeople.setStyleSheet(
                "border: 1px solid white; color: white; font-weight: bold; font-size: 14px; qproperty-alignment: AlignCenter;"
            )

    def update_tag_status(self, tag_index, status):
        """
        태그의 상태를 업데이트하고 wsLog에 출력합니다.
        """
        if tag_index >= len(self.tag_status_list):
            return

        if self.tag_status_list[tag_index] == status:
            return  # 변경이 없으면 업데이트하지 않음

        self.tag_status_list[tag_index] = status  # 상태 업데이트
        tag_info = self.tag_names.get(f"Tag {tag_index}", f"Tag {tag_index}")
        tag_name = tag_info.get("tagName")

    # ///////////////////////////////////////////////////////////////
    # 앵커 관련
    # ///////////////////////////////////////////////////////////////

    # 앵커 수에 따라 Kalman Filters를 초기화.
    def initialize_kalman_filters(self):
        anchor_count = self.parent.ui.g_anchorNum.value()
        self.calculation = Calculation(anchor_count)  # Offsets 전달

    # 앵커 신호 따라 색상 변경(회색/파랑)
    def update_anchor_status(self, range_values, rssi_values, anchor_count):
        for i in range(anchor_count):
            range_val = range_values[i]
            rssi_val = rssi_values[i]

            anchor_name = f"Anchor {i}"  # 현재 앵커 이름

            if anchor_name in self.anchor_labels:
                anchor_widget = self.anchor_labels[anchor_name]
                image_label = anchor_widget.findChild(QLabel)

                if image_label:
                    if range_val == 0 and rssi_val == 0:
                        image_path = resource_path("modules/anchor_off.png")
                    else:
                        image_path = resource_path("modules/anchor.png")

                    pixmap = QPixmap(image_path).scaled(30, 30)
                    image_label.setPixmap(pixmap)

    # ///////////////////////////////////////////////////////////////
    # 워크스페이스 관련 설정
    # ///////////////////////////////////////////////////////////////

    # 작업 공간 정보 가져오기 함수
    def draw_workspace_box(self, x, y, workspace_width, workspace_height, anchors, vertices):
        # QFrame 크기 가져오기
        frame_width = self.parent.ui.workspace.width()
        frame_height = self.parent.ui.workspace.height()

        # 작업 공간 크기가 0인 경우 그리지 않음
        if frame_width == 0 or frame_height == 0:
            return

        # 스케일 비율 계산 (작업 공간을 QFrame에 맞게 조정)
        self.scale_ratio = min(frame_width / workspace_width, frame_height / workspace_height) * 0.9

        # 작업 공간 크기 조정
        scaled_width = workspace_width * self.scale_ratio
        scaled_height = workspace_height * self.scale_ratio

        # 중앙 정렬을 위한 offset 계산
        x_offset = (frame_width - scaled_width) / 2
        y_offset = (frame_height - scaled_height) / 2

        # 작업 공간 박스 설정
        self.workspace_box = QRectF(x_offset, y_offset, scaled_width, scaled_height)

        # 앵커 위치 스케일링 및 표시
        for anchor_name, coordinates in anchors.items():
            anchor_x_scaled = (coordinates["x"] / workspace_width) * scaled_width + x_offset - 15
            anchor_y_scaled = (coordinates["y"] / workspace_height) * scaled_height + y_offset - 15

            # 앵커 라벨 표시
            label = self.anchor_labels.get(anchor_name)
            if label:
                label.move(int(anchor_x_scaled), int(anchor_y_scaled))
                label.show()

        # Vertex 점 리스트 초기화
        self.vertex_points = []

        # Vertex 위치를 스케일링하여 리스트에 저장
        for vertex_name, coordinates in vertices.items():
            vertex_x_scaled = (coordinates["x"] / workspace_width) * scaled_width + x_offset
            vertex_y_scaled = (coordinates["y"] / workspace_height) * scaled_height + y_offset

            # Vertex 점 저장 (QPointF 사용)
            self.vertex_points.append((vertex_x_scaled, vertex_y_scaled))

        # QFrame 다시 그리기
        self.parent.ui.workspace.update()

    # 초기화 후에 다시 작업 공간 그리기
    def redraw_workspace_after_init(self):
        if self.workspace_loaded and hasattr(self, 'parent'):
            settings = self.parent.workspace_settings
            self.draw_workspace_box(
                x=0,
                y=0,
                workspace_width=settings.get("workspace_width", 0),
                workspace_height=settings.get("workspace_height", 0),
                anchors=self.anchor_data,
                vertices=self.vertex_data
            )
            self.parent.ui.workspace.update()

    # 작업 공간, 위험 구역, 태그 그리기
    def paint_workspace(self, event):
        painter = QPainter(self.parent.ui.workspace)
        painter.setRenderHint(QPainter.Antialiasing)

        # 작업 공간 박스 그리기
        if hasattr(self, "workspace_box") and self.workspace_box:
            painter.setBrush(self.workspace_color)  # 작업 공간 색상
            painter.drawRect(self.workspace_box)

        #  여기서 PNG를 vertex 폴리곤에 맞춰 그림
        self._draw_png_in_vertices(painter)

        # Vertex 점 그리기 및 선 연결
        if not hasattr(self, "vertex_points") or not self.vertex_points:
            return  # 그릴 꼭짓점이 없으면 그리기 생략

        if hasattr(self, "vertex_points"):
            preview_points = self.vertex_points.copy()

            # 만약 이동 중인 vertex가 있다면, 해당 인덱스만 임시로 좌표 변경
            if hasattr(self, "moving_vertex") and self.moving_vertex and self.preview_vertex_position:
                try:
                    index = int(self.moving_vertex.split(" ")[1]) - 1  # "Vertex 3" -> 2
                    preview_points[index] = self.preview_vertex_position
                except Exception as e:
                    print(f"[WARNING] 미리보기 vertex 좌표 설정 오류: {e}")

            # 선 그리기
            if len(preview_points) > 1:
                pen = QPen(self.danger_color, 4)
                painter.setPen(pen)
                for i in range(len(preview_points) - 1):
                    x1, y1 = preview_points[i]
                    x2, y2 = preview_points[i + 1]
                    painter.drawLine(int(x1), int(y1), int(x2), int(y2))
                # 처음과 마지막 연결
                x_first, y_first = preview_points[0]
                x_last, y_last = preview_points[-1]
                painter.drawLine(int(x_last), int(y_last), int(x_first), int(y_first))

        # Vertex 점 및 이름 표시
        for i, (x, y) in enumerate(self.vertex_points):
            # Vertex 점 (노란색 원)
            painter.setBrush(QColor(255, 255, 0))  # 노란색
            painter.setPen(Qt.NoPen)  # 내부 채우기만 적용
            painter.drawEllipse(int(x) - 5, int(y) - 5, 10, 10)

            # Vertex 이름 ("V1", "V2" ...)
            painter.setPen(Qt.white)  # 흰색 글자
            painter.setFont(QFont("Arial", 10, QFont.Bold))
            painter.drawText(int(x) + 5, int(y) - 5, f"V{i + 1}")

        # 태그 그리기 (inactive 상태면 그리지 않음)
        if hasattr(self, "tag_positions") and hasattr(self, "tag_status_list"):
            for tag_index, (x, y) in self.tag_positions.items():
                # tag_status_list가 리스트라면 tag_index 범위 체크
                if isinstance(self.tag_status_list, list) and tag_index < len(self.tag_status_list):
                    if self.tag_status_list[tag_index] == "inactive":
                        continue  # inactive 상태면 그리지 않음

                # 태그 원 그리기
                painter.setPen(Qt.NoPen)
                painter.setBrush(QColor(255, 187, 63))
                painter.drawEllipse(x - 5, y - 5, 15, 15)

                # 태그 이름 (태그 원 아래에 표시)
                painter.setPen(Qt.white)
                painter.setFont(QFont("Arial", 9, QFont.Bold))
                name_to_show = f"Tag {tag_index}"
                painter.drawText(int(x - 15), int(y + 25), name_to_show)

    # ///////////////////////////////////////////////////////////////
    # workspace 파일 관련 설정
    # ///////////////////////////////////////////////////////////////

    # 1. open 버튼 클릭 시 실행
    def open_existing_workspace(self):
        print("open workspace")
        workspace_list = self.get_workspace_list()

        # 보조 창 생성
        dialog = QDialog(self.parent)
        dialog.setWindowTitle("워크스페이스 선택")
        dialog.setFixedSize(400, 300)

        # 리스트 위젯
        list_widget = QListWidget(dialog)
        list_widget.addItems(workspace_list)

        # 확인 버튼
        select_button = QPushButton("선택한 워크스페이스 로드", dialog)
        select_button.pressed.connect(lambda: self.load_selected_workspace(dialog, list_widget))

        # 삭제 버튼
        delete_button = QPushButton("선택한 워크스페이스 삭제", dialog)
        delete_button.pressed.connect(lambda: self.delete_selected_workspace(dialog, list_widget))

        # 이름 변경 버튼
        rename_button = QPushButton("선택한 워크스페이스 이름 변경", dialog)
        rename_button.pressed.connect(lambda: self.rename_selected_workspace(dialog, list_widget))

        # 내보내기 버튼
        export_button = QPushButton("워크스페이스 내보내기", dialog)
        export_button.pressed.connect(lambda: self.export_selected_workspace(list_widget))

        # 레이아웃 설정
        layout = QVBoxLayout(dialog)
        layout.addWidget(list_widget)
        layout.addWidget(select_button)
        layout.addWidget(delete_button)
        layout.addWidget(rename_button)
        layout.addWidget(export_button)

        dialog.setLayout(layout)

        # 창 띄우기
        dialog.exec()

    # 1-1. 선택한 워크스페이스 로드
    def load_selected_workspace(self, dialog, list_widget):
        selected_item = list_widget.currentItem()
        if not selected_item:
            QMessageBox.warning(self.parent, "Warning", "Please select a workspace!")
            return

        workspace_name = selected_item.text()
        data = self.load_workspace_from_db(workspace_name)

        if data:
            self.current_workspace_name = workspace_name
            self.apply_workspace_data(data)

            QMessageBox.information(self.parent, "Success", f"'{workspace_name}' 워크스페이스가 성공적으로 로드되었습니다.")
            self.update_current_workspace(workspace_name)
            dialog.accept()  # 보조 창 닫기
        else:
            QMessageBox.critical(self.parent, "Error", f"'{workspace_name}'워크스페이스 로드에 실패했습니다.")
            self.current_workspace_name = None  # 초기화

    # 1-2. 선택한 워크스페이스 삭제
    def delete_selected_workspace(self, dialog, list_widget):
        selected_item = list_widget.currentItem()
        if not selected_item:
            QMessageBox.warning(self.parent, "Warning", "Please select a workspace to delete!")
            return

        workspace_name = selected_item.text()

        confirm = QMessageBox.question(
            self.parent,
            "삭제 확인",
            f"'{workspace_name}' 워크스페이스를 삭제하시겠습니까?",
            QMessageBox.Yes | QMessageBox.No
        )

        if confirm == QMessageBox.Yes:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            try:
                cursor.execute("DELETE FROM workspaces WHERE name = ?", (workspace_name,))
                conn.commit()
                QMessageBox.information(self.parent, "Deleted", f"'{workspace_name}' 워크스페이스가 삭제되었습니다.")

                # 리스트에서 삭제한 항목 제거
                list_widget.takeItem(list_widget.row(selected_item))

                # current_workspace_name도 초기화할지 선택적으로 처리 가능
                if self.current_workspace_name == workspace_name:
                    self.current_workspace_name = None

            except sqlite3.Error as e:
                QMessageBox.critical(self.parent, "Error", f"삭제 실패했습니다: {e}")
            finally:
                conn.close()

    # 1-3. 선택한 워크스페이스 이름 변경
    def rename_selected_workspace(self, dialog, list_widget):
        selected_item = list_widget.currentItem()
        if not selected_item:
            QMessageBox.warning(self.parent, "경고", "이름을 변경할 워크스페이스를 선택해주세요.")
            return

        old_name = selected_item.text()

        new_name, ok = QInputDialog.getText(self.parent, "워크스페이스 이름 변경", "새 워크스페이스 이름을 입력해주세요:")
        if not ok or not new_name.strip():
            return  # 취소 또는 빈 문자열 입력 시 종료

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        try:
            # 이름 중복 확인
            cursor.execute("SELECT COUNT(*) FROM workspaces WHERE name = ?", (new_name,))
            if cursor.fetchone()[0] > 0:
                QMessageBox.warning(self.parent, "중복된 이름", f"'{new_name}'이라는 이름의 워크스페이스가 이미 존재합니다.")
                return

            # 이름 변경
            cursor.execute("UPDATE workspaces SET name = ? WHERE name = ?", (new_name, old_name))
            conn.commit()

            QMessageBox.information(self.parent, "이름 변경 완료", f"워크스페이스 이름이 '{old_name}'에서 '{new_name}'(으)로 변경되었습니다.")
            selected_item.setText(new_name)

            if self.current_workspace_name == old_name:
                self.current_workspace_name = new_name
                self.edit_workspace()

        except sqlite3.Error as e:
            QMessageBox.critical(self.parent, "오류", f"이름 변경 중 오류가 발생했습니다:\n{e}")
        finally:
            conn.close()

    # 1-4. 선택한 워크스페이스 내보내기
    def export_selected_workspace(self, list_widget):
        selected_item = list_widget.currentItem()
        if not selected_item:
            QMessageBox.warning(self.parent, "경고", "내보낼 워크스페이스를 선택해주세요.")
            return

        workspace_name = selected_item.text()

        # 드라이브 선택 다이얼로그 호출
        selected_drive = show_drive_select_dialog(self.parent)

        if not selected_drive:
            QMessageBox.warning(self.parent, "경고", "드라이브를 선택해주세요.")
            return

        # 선택한 드라이브 루트에 저장
        file_path = os.path.join(selected_drive, f"{workspace_name}.db")

        try:
            # 기존 DB에서 선택된 워크스페이스 한 줄 가져오기
            conn_src = sqlite3.connect(self.db_path)
            cursor_src = conn_src.cursor()
            cursor_src.execute("""
                    SELECT id, name, data, current, image_path, image_width, image_height, image_offset_x, image_offset_y 
                    FROM workspaces 
                    WHERE name = ?
                """, (workspace_name,))
            row = cursor_src.fetchone()
            conn_src.close()

            if not row:
                QMessageBox.critical(self.parent, "오류", "선택한 워크스페이스 데이터를 찾을 수 없습니다.")
                return

            # 새 DB에 저장
            conn_dst = sqlite3.connect(file_path)
            cursor_dst = conn_dst.cursor()
            cursor_dst.execute("""
                    CREATE TABLE IF NOT EXISTS workspaces (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        name TEXT UNIQUE,
                        data TEXT,
                        current INTEGER DEFAULT 0,
                        image_path TEXT,
                        image_width REAL,
                        image_height REAL,
                        image_offset_x REAL,
                        image_offset_y REAL
                    )
                """)
            cursor_dst.execute("""
                    INSERT INTO workspaces (
                        id, name, data, current,
                        image_path, image_width, image_height, image_offset_x, image_offset_y
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, row)
            conn_dst.commit()
            conn_dst.close()

            QMessageBox.information(self.parent, "성공", f"워크스페이스가 '{file_path}'로 내보내졌습니다.")
        except Exception as e:
            QMessageBox.critical(self.parent, "오류", f"내보내기 중 오류가 발생했습니다:\n{e}")

    # 2. 새로운 워크스페이스로 저장
    def save_as_new_workspace(self):
        workspace_name, ok = QInputDialog.getText(self.parent, "Save Workspace", "Enter workspace name:")
        if not ok or not workspace_name.strip():
            QMessageBox.warning(self.parent, "Warning", "Workspace name cannot be empty.")
            return

        # 워크스페이스 데이터 생성
        workspace_data = {
            "workspace_settings": {
                "workspace_width": self.parent.ui.a_workspace_width.value(),
                "workspace_height": self.parent.ui.b_workspace_height.value(),
            },
            "vertexData": self.vertex_data,
            "vertex_count": self.parent.ui.vertexCount.value(),
            "anchors": self.anchor_data,
            "anchor_count": self.parent.ui.g_anchorNum.value(),
            "tag_count": self.parent.ui.h_tagNum.value(),
            "tags": {
                f"Tag {i}": self.tag_names.get(f"Tag {i}", {
                    "tagName": "None",
                    "RFIDid": "None",
                    "states": 0,
                }) for i in range(self.parent.ui.h_tagNum.value())
            }
        }

        # 데이터베이스에 저장
        self.save_workspace_to_db(workspace_name, workspace_data)

        # 현재 워크스페이스 이름 업데이트
        self.current_workspace_name = workspace_name
        self.update_current_workspace(workspace_name)

        data = self.load_workspace_from_db(workspace_name)
        if data:
            self.apply_workspace_data(data)

    def apply_workspace_data(self, data):
        """
        UI에 워크스페이스 데이터를 적용하고 QFrame에 그림을 그림
        """
        workspace_settings = data.get("workspace_settings", {})
        self.parent.workspace_settings = workspace_settings

        self.parent.ui.a_workspace_width.setValue(workspace_settings.get("workspace_width", 0))
        self.parent.ui.b_workspace_height.setValue(workspace_settings.get("workspace_height", 0))
        self.parent.ui.g_anchorNum.setValue(data.get("anchor_count", 0))
        self.parent.ui.vertexCount.setValue(data.get("vertex_count", 0))
        self.parent.ui.h_tagNum.setValue(data.get("tag_count", 0))

        # 앵커 데이터 업데이트
        self.anchor_data = data.get("anchors", {})
        self.update_visible_anchors()
        self.update_anchor_positions()

        # 꼭짓점 데이터 업데이트
        self.vertex_data = data.get("vertexData", {})
        self.update_vertex_list()
        self.parent.ui.vertexCount.valueChanged.connect(self.update_vertex_list)

        # 첫 번째 앵커를 선택하고 X, Y 좌표 업데이트
        if self.anchor_data:
            first_anchor = list(self.anchor_data.keys())[0]
            self.parent.ui.i_anchorSelect.setCurrentText(first_anchor)
            self.update_anchor_position()  # SpinBox 값 갱신

        # 첫 번째 꼭짓점을 선택하고 X, Y 좌표 업데이트
        if self.vertex_data:
            first_vertex = list(self.vertex_data.keys())[0]
            self.parent.ui.vertexSelect.setCurrentText(first_vertex)
            self.update_vertex_position()  # SpinBox 값 갱신

        # 작업 공간 로드 상태 설정
        self.workspace_loaded = True

        # QFrame에 그림을 그리기 위해 draw_workspace_box 호출
        self.draw_workspace_box(
            x=0, y=0,
            workspace_width=workspace_settings.get("workspace_width", 0),
            workspace_height=workspace_settings.get("workspace_height", 0),
            anchors=self.anchor_data,
            vertices=self.vertex_data
        )
        self.parent.ui.workspace.update()  # 테두리 포함 다시 그리기

        self.set_background_png(r"C:\Users\DS\Downloads\u_guide_0902\u_guide_0902\modules\subway.png", opacity=0.85)

        self._build_or_update_intent_runtime()

        if hasattr(self.parent.ui, "workspaceName"):
            self.parent.ui.workspaceName.setText(f"{self.current_workspace_name}")

    # 3. 워크스페이스 수정
    def edit_workspace(self):
        if not hasattr(self, "current_workspace_name") or not self.current_workspace_name:
            QMessageBox.warning(self.parent, "Warning", "현재 열려 있는 워크스페이스가 없습니다.")
            return

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        try:
            # 기존 데이터 조회
            cursor.execute("SELECT current FROM workspaces WHERE name = ?", (self.current_workspace_name,))
            row = cursor.fetchone()

            if row:
                current = row[0]
            else:
                current = 0

            # 수정된 워크스페이스 데이터 생성
            workspace_data = {
                "workspace_settings": {
                    "workspace_width": self.parent.ui.a_workspace_width.value(),
                    "workspace_height": self.parent.ui.b_workspace_height.value(),
                },
                "vertexData": self.vertex_data,  # 꼭짓점 데이터 추가
                "vertex_count": self.parent.ui.vertexCount.value(),
                "anchors": self.anchor_data,
                "anchor_count": self.parent.ui.g_anchorNum.value(),
                "tag_count": self.parent.ui.h_tagNum.value(),
                "tags": {
                    f"Tag {i}": self.tag_names.get(f"Tag {i}", {
                        "tagName": "None",
                        "RFIDid": "None",
                        "states": 0
                    }) for i in range(self.parent.ui.h_tagNum.value())
                }

            }

            # 데이터베이스 업데이트
            cursor.execute("""
                  UPDATE workspaces
                  SET data = ?, current = ?
                  WHERE name = ?
              """, (json.dumps(workspace_data), current, self.current_workspace_name))

            conn.commit()

            # Update the workspace drawings
            self.apply_workspace_data(workspace_data)  # Apply changes to UI
            QMessageBox.information(self.parent, "Success",
                                    f"'{self.current_workspace_name}' 공간 수정이 완료되었습니다.")
        except sqlite3.Error as e:
            print(f"[ERROR] 워크스페이스 수정 실패: {e}")
        finally:
            conn.close()

    def update_vertex_list(self):
        """
        vertexCount 값이 변경될 때 vertexSelect 콤보박스를 업데이트합니다.
        """
        count = self.parent.ui.vertexCount.value()  # 현재 vertexCount 값 가져오기

        # 꼭짓점 개수 업데이트 (부족하면 추가, 초과하면 삭제)
        for i in range(count):
            vertex_name = f"Vertex {i + 1}"
            if vertex_name not in self.vertex_data:
                # 작업 공간 크기 기준
                workspace_width = self.parent.ui.a_workspace_width.value()
                workspace_height = self.parent.ui.b_workspace_height.value()

                # 안전 여백 확보
                margin = 1.0

                # X 좌표: 정해진 간격으로 나열 (width 범위 내)
                spacing = (workspace_width - 2 * margin) / max(count, 1)
                x = margin + spacing * i

                # Y 좌표: 고정값 (위쪽에서 margin 만큼 떨어진 곳)
                y = margin

                # 저장
                self.vertex_data[vertex_name] = {"x": round(x, 2), "y": round(y, 2)}

        # 불필요한 꼭짓점 제거
        existing_keys = list(self.vertex_data.keys())
        for key in existing_keys:
            index = int(key.split(" ")[1]) - 1  # "Vertex 1" -> 0
            if index >= count:
                del self.vertex_data[key]

        # 콤보박스 업데이트
        self.parent.ui.vertexSelect.clear()
        self.parent.ui.vertexSelect.addItems(self.vertex_data.keys())

        # 첫 번째 꼭짓점 자동 선택
        if self.vertex_data:
            first_vertex = list(self.vertex_data.keys())[0]
            self.parent.ui.vertexSelect.setCurrentText(first_vertex)
            self.update_vertex_position()  # SpinBox 값 갱신

    def update_vertex_position(self):
        """
        선택한 꼭짓점의 x, y 좌표를 SpinBox에 반영합니다.
        """
        selected_vertex = self.parent.ui.vertexSelect.currentText()
        if not selected_vertex.strip():
            return

        if selected_vertex not in self.vertex_data:
            self.vertex_data[selected_vertex] = {"x": 0.0, "y": 0.0}

        x = self.vertex_data[selected_vertex]["x"]
        y = self.vertex_data[selected_vertex]["y"]

        # SpinBox 업데이트 (이벤트 중복 방지)
        self.parent.ui.vertexX.blockSignals(True)
        self.parent.ui.vertexY.blockSignals(True)
        self.parent.ui.vertexX.setValue(x)
        self.parent.ui.vertexY.setValue(y)
        self.parent.ui.vertexX.blockSignals(False)
        self.parent.ui.vertexY.blockSignals(False)

    def save_vertex_position(self):
        """
        SpinBox에서 변경된 x, y 값을 vertex_data에 저장합니다.
        """
        selected_vertex = self.parent.ui.vertexSelect.currentText()
        if not selected_vertex.strip():
            return

        if selected_vertex in self.vertex_data:
            self.vertex_data[selected_vertex]["x"] = self.parent.ui.vertexX.value()
            self.vertex_data[selected_vertex]["y"] = self.parent.ui.vertexY.value()

    def update_anchor_position(self):
        """
        콤보박스에서 선택된 앵커의 X, Y 좌표를 SpinBox에 업데이트합니다.
        """
        # 콤보박스에서 선택된 앵커 이름 가져오기
        selected_anchor = self.parent.ui.i_anchorSelect.currentText()

        # 빈 문자열 제외
        if not selected_anchor.strip():
            return

        # 선택된 앵커가 anchor_data에 없으면 초기화 (값 0, 0으로 설정)
        if selected_anchor not in self.anchor_data:
            self.anchor_data[selected_anchor] = {"x": 0.0, "y": 0.0}

        # 선택된 앵커의 좌표 가져오기
        x = self.anchor_data[selected_anchor]["x"]
        y = self.anchor_data[selected_anchor]["y"]

        # SpinBox에 값 설정 (신호 차단으로 이벤트 중복 방지)
        self.parent.ui.j_anchorX.blockSignals(True)
        self.parent.ui.k_anchorY.blockSignals(True)
        self.parent.ui.j_anchorX.setValue(x)
        self.parent.ui.k_anchorY.setValue(y)
        self.parent.ui.j_anchorX.blockSignals(False)
        self.parent.ui.k_anchorY.blockSignals(False)

        # self.anchor_positions에도 업데이트
        try:
            # selected_anchor의 인덱스를 추출
            anchor_index = list(self.anchor_data.keys()).index(selected_anchor)

            # anchor_positions 크기를 동기화
            if len(self.anchor_positions) <= anchor_index:
                self.anchor_positions.extend([(0.0, 0.0)] * (anchor_index + 1 - len(self.anchor_positions)))

            # 현재 선택된 앵커의 좌표를 업데이트
            self.anchor_positions[anchor_index] = (x, y)
            # print(f"[DEBUG] Updated self.anchor_positions[{anchor_index}] to: {self.anchor_positions[anchor_index]}")
        except ValueError:
            print(f"[WARNING] Selected anchor '{selected_anchor}' not found in anchor_data.")

    def save_anchor_position(self):
        """
        SpinBox에서 입력된 X, Y 좌표를 현재 선택된 앵커에 저장합니다.
        """
        selected_anchor = self.parent.ui.i_anchorSelect.currentText()

        if selected_anchor in self.anchor_data:
            # SpinBox 값을 가져와서 저장
            x = self.parent.ui.j_anchorX.value()
            y = self.parent.ui.k_anchorY.value()
            self.anchor_data[selected_anchor]["x"] = x
            self.anchor_data[selected_anchor]["y"] = y

            # 앵커 위치 업데이트
            self.update_anchor_positions()

    # 앵커 표시 업데이트
    def update_visible_anchors(self):
        anchor_count = self.parent.ui.g_anchorNum.value()  # 표시할 앵커 수
        current_count = len(self.anchor_labels)

        # 앵커 추가
        if anchor_count > current_count:
            for i in range(current_count, anchor_count):
                anchor_name = f"Anchor {i}"
                # anchor_data에 기본값 추가
                self.anchor_data[anchor_name] = {"x": 0.0, "y": 0.0}

                # QLabel과 레이아웃 생성
                anchor_widget = QWidget(self.parent.ui.workspace)  # 하나의 위젯에 이미지와 텍스트 포함
                anchor_widget.setStyleSheet("border: none; background-color: transparent;")
                layout = QVBoxLayout(anchor_widget)
                layout.setContentsMargins(0, 0, 0, 0)  # 여백 제거
                layout.setSpacing(5)  # 이미지와 텍스트 사이 간격

                # 이미지 QLabel 생성
                image_label = QLabel(anchor_widget)
                pixmap = QPixmap(resource_path("modules/anchor_off.png"))
                if not pixmap.isNull():
                    scaled_pixmap = pixmap.scaled(30, 30)  # 이미지 크기 조정
                    image_label.setPixmap(scaled_pixmap)
                    image_label.setAlignment(Qt.AlignCenter)

                # 텍스트 QLabel 생성
                text_label = QLabel(anchor_name, anchor_widget)
                text_label.setStyleSheet("font-size: 10px; color: white; font-weight: bold;")
                text_label.setAlignment(Qt.AlignCenter)

                # 레이아웃에 추가
                layout.addWidget(image_label)
                layout.addWidget(text_label)
                anchor_widget.setLayout(layout)

                # anchor_labels에 추가
                self.anchor_labels[anchor_name] = anchor_widget

                # UI에 표시
                anchor_widget.show()

        # 앵커 좌표 스케일링 및 QLabel 이동
        for anchor_name, coords in self.anchor_data.items():
            if anchor_name in self.anchor_labels:
                anchor_widget = self.anchor_labels[anchor_name]

                # 스케일링 적용
                if hasattr(self, "scale_ratio") and hasattr(self, "workspace_box"):
                    x_scaled = coords["x"] * self.scale_ratio + self.workspace_box.x() - 15
                    y_scaled = coords["y"] * self.scale_ratio + self.workspace_box.y() - 15
                    # QLabel 이동
                    anchor_widget.move(int(x_scaled), int(y_scaled))

    def initialize_anchor_labels(self):
        """
        화면 상에 앵커를 표시하기 위해 QLabel을 초기화합니다.
        기존 앵커 위치는 유지하고, 추가된 앵커만 새로 생성합니다.
        """
        # 기존 anchor_labels에 없는 라벨만 새로 생성
        anchor_count = self.parent.ui.g_anchorNum.value()

        # 1. anchor_data에 기존 값은 유지하고, 부족한 항목만 추가
        for i in range(anchor_count):
            anchor_name = f"Anchor {i}"
            if anchor_name not in self.anchor_data:
                self.anchor_data[anchor_name] = {"x": 0.0, "y": 0.0}

        # 2. anchor_data 중 개수를 초과한 것 삭제
        to_delete = [k for k in self.anchor_data.keys() if int(k.split(" ")[1]) >= anchor_count]
        for k in to_delete:
            self.anchor_data.pop(k)

        # 3. 기존 anchor_labels는 삭제하지 않고, 부족한 라벨만 추가
        for i in range(anchor_count):
            anchor_name = f"Anchor {i}"
            if anchor_name not in self.anchor_labels:
                anchor_widget = QWidget(self.parent.ui.workspace)
                anchor_widget.setStyleSheet("border: none; background-color: transparent;")
                layout = QVBoxLayout(anchor_widget)
                layout.setContentsMargins(0, 0, 0, 0)
                layout.setSpacing(5)

                image_label = QLabel(anchor_widget)
                pixmap = QPixmap(resource_path("modules/anchor_off.png")).scaled(30, 30)
                image_label.setPixmap(pixmap)
                image_label.setAlignment(Qt.AlignCenter)

                text_label = QLabel(anchor_name, anchor_widget)
                text_label.setStyleSheet("font-size: 10px; color: white; font-weight: bold;")
                text_label.setAlignment(Qt.AlignCenter)

                layout.addWidget(image_label)
                layout.addWidget(text_label)
                anchor_widget.setLayout(layout)

                self.anchor_labels[anchor_name] = anchor_widget
                anchor_widget.show()

        # 4. anchor_labels 중 초과된 것은 제거
        to_remove = [name for name in self.anchor_labels if int(name.split(" ")[1]) >= anchor_count]
        for name in to_remove:
            self.anchor_labels[name].deleteLater()
            del self.anchor_labels[name]

        # 5. self.anchor_positions도 동기화
        self.anchor_positions = [
            (self.anchor_data[f"Anchor {i}"]["x"], self.anchor_data[f"Anchor {i}"]["y"])
            if f"Anchor {i}" in self.anchor_data else (0.0, 0.0)
            for i in range(anchor_count)
        ]

    def update_anchor_settings(self):
        # 앵커 개수를 가져옴
        anchor_count = self.parent.ui.g_anchorNum.value()

        self.parent.anchor_data = {
            f"Anchor {i}": {"x": 0.0, "y": 0.0} for i in range(anchor_count)
        }

        self.parent.ui.i_anchorSelect.clear()

        # 앵커 개수에 따라 항목 추가
        for i in range(anchor_count):
            self.parent.ui.i_anchorSelect.addItem(f"Anchor {i}")

    def update_anchor_positions(self):
        """
        self.anchor_data를 기반으로 self.anchor_positions 업데이트.
        """
        try:
            self.anchor_positions = []  # 앵커 위치 리스트 초기화
            for anchor_name, coords in self.anchor_data.items():
                if "x" in coords and "y" in coords:
                    self.anchor_positions.append((coords["x"], coords["y"]))
        except Exception as e:
            print(f"[ERROR] Failed to update anchor positions: {e}")

    def update_anchor_count(self):
        """
        g_anchorNum 값 변경 시 호출. 앵커 데이터를 업데이트.
        """
        try:
            # 앵커 라벨 및 데이터 초기화
            self.initialize_anchor_labels()
            self.update_anchor_positions()

        except Exception as e:
            print(f"Error updating anchor count: {e}")

    # ///////////////////////////////////////////////////////////////
    # 데이터베이스 관련
    # ///////////////////////////////////////////////////////////////
    def initialize_database(self):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # 데이터베이스에 테이블 생성
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS workspaces (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT UNIQUE,
            data TEXT
            current INTEGER DEFAULT 0
        )
        """)

        # 새로운 컬럼 추가
        try:
            cursor.execute("ALTER TABLE workspaces ADD COLUMN image_path TEXT")
        except sqlite3.OperationalError:
            pass
        try:
            cursor.execute("ALTER TABLE workspaces ADD COLUMN image_width REAL DEFAULT 0")
        except sqlite3.OperationalError:
            pass
        try:
            cursor.execute("ALTER TABLE workspaces ADD COLUMN image_height REAL DEFAULT 0")
        except sqlite3.OperationalError:
            pass

        conn.commit()
        conn.close()
        # print("[INFO] Database initialized!")

    # 데이터베이스에 워크스페이스 저장
    def save_workspace_to_db(self, name, data):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        try:
            # 기존 워크스페이스 데이터 조회
            cursor.execute("SELECT current FROM workspaces WHERE name = ?", (name,))
            row = cursor.fetchone()

            # 기존 값 유지
            if row:
                current = row[0]
            else:
                current = 0  # 기본값

            # INSERT 또는 UPDATE 시 기존 값 유지
            cursor.execute("""
                INSERT INTO workspaces (name, data, current)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(name) DO UPDATE SET
                    data = excluded.data,
                    current = COALESCE(excluded.current, workspaces.current)
            """, (name, json.dumps(data), current))

            conn.commit()
            # print(f"[INFO] Workspace '{name}' saved to database.")
        except sqlite3.Error as e:
            print(f"[ERROR] Failed to save workspace: {e}")
        finally:
            conn.close()

    # 데이터베이스에서 워크스페이스 가져오기
    def load_workspace_from_db(self, name):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        try:
            cursor.execute("SELECT data FROM workspaces WHERE name = ?", (name,))
            row = cursor.fetchone()
            if row:
                data = json.loads(row[0])  # JSON 데이터를 파싱
                # print(f"[DEBUG] Loaded workspace data: {data}")  # 디버깅 출력
                return data
            else:
                # print(f"[WARNING] Workspace '{name}' not found in database.")
                return None
        except sqlite3.Error as e:
            # print(f"[ERROR] Failed to load workspace: {e}")
            return None
        finally:
            conn.close()

    def get_workspace_list(self):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        try:
            cursor.execute("SELECT name FROM workspaces")
            rows = cursor.fetchall()
            return [row[0] for row in rows]
        except sqlite3.Error as e:
            # print(f"[ERROR] Failed to fetch workspace list: {e}")
            return []
        finally:
            conn.close()

    def load_last_workspace(self):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        try:
            # 전체 워크스페이스 개수 확인
            cursor.execute("SELECT COUNT(*) FROM workspaces")
            total_count = cursor.fetchone()[0]

            # 워크스페이스가 1개뿐이면 그것을 불러옴
            if total_count == 1:
                cursor.execute("SELECT name FROM workspaces")
            else:
                # 여러 개면 current=1인 것 우선 시도
                cursor.execute("SELECT name FROM workspaces WHERE current = 1")

            row = cursor.fetchone()

            if row:
                last_workspace_name = row[0]
                data = self.load_workspace_from_db(last_workspace_name)
                if data:
                    self.current_workspace_name = last_workspace_name
                    self.apply_workspace_data(data)
                    # print(f"[INFO] '{last_workspace_name}' 워크스페이스 자동 로드 완료.")
                else:
                    # print("[WARNING] 워크스페이스 데이터를 찾을 수 없습니다.")
                    self.current_workspace_name = None
                    self.workspace_loaded = False
            else:
                # print("[INFO] 불러올 워크스페이스가 없습니다.")
                self.current_workspace_name = None
                self.workspace_loaded = False
        finally:
            conn.close()

    def update_current_workspace(self, workspace_name):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        try:
            # 모든 워크스페이스의 current 값을 0으로 초기화
            cursor.execute("UPDATE workspaces SET current = 0")
            # 선택한 워크스페이스의 current 값을 1로 설정
            cursor.execute("UPDATE workspaces SET current = 1 WHERE name = ?", (workspace_name,))
            conn.commit()
            # print(f"[INFO] Updated current workspace to: {workspace_name}")
        finally:
            conn.close()

    # ///////////////////////////////////////////////////////////////
    # 좌클릭 통한 vertex, anchor 위치 조정
    # ///////////////////////////////////////////////////////////////
    def handle_workspace_click(self, event):
        if event.button() != Qt.LeftButton:
            return

        click_pos = event.position()

        # 스케일 관련 정보 없으면 종료
        if not hasattr(self, "scale_ratio") or not hasattr(self, "workspace_box"):
            return

        # 실좌표 계산
        x_real = (click_pos.x() - self.workspace_box.x()) / self.scale_ratio
        y_real = (click_pos.y() - self.workspace_box.y()) / self.scale_ratio

        # (1) 이동 확정 중이면 여기서 처리
        if hasattr(self, "moving_anchor") and self.moving_anchor:
            self.confirm_anchor_position(x_real, y_real)
            return

        if hasattr(self, "moving_vertex") and self.moving_vertex:
            self.confirm_vertex_position(x_real, y_real)
            return

        # (2) 선택 가능한 요소가 있는지 확인
        clicked_vertex = self.find_clicked_vertex(click_pos)
        clicked_anchor = self.find_clicked_anchor(click_pos)

        if clicked_vertex and clicked_anchor:
            # 겹친 경우 선택 팝업 (사용자 정의 버튼)
            msg_box = QMessageBox(self.parent)
            msg_box.setWindowTitle("선택 대상 확인")
            msg_box.setText("이 위치에 Anchor와 Vertex가 모두 있습니다.\n무엇을 이동하시겠습니까?")

            anchor_button = msg_box.addButton("앵커", QMessageBox.ActionRole)
            vertex_button = msg_box.addButton("꼭짓점", QMessageBox.ActionRole)
            cancel_button = msg_box.addButton("취소", QMessageBox.RejectRole)

            msg_box.exec()

            clicked_button = msg_box.clickedButton()
            if clicked_button == anchor_button:
                self.moving_anchor = clicked_anchor
                self.parent.ui.workspace.setCursor(Qt.CrossCursor)
            elif clicked_button == vertex_button:
                self.moving_vertex = clicked_vertex
                self.parent.ui.workspace.setCursor(Qt.CrossCursor)
            return

        elif clicked_anchor:
            self.moving_anchor = clicked_anchor
            self.parent.ui.workspace.setCursor(Qt.CrossCursor)
            return

        elif clicked_vertex:
            self.moving_vertex = clicked_vertex
            self.parent.ui.workspace.setCursor(Qt.CrossCursor)
            return

    def find_clicked_vertex(self, pos):
        """ 클릭 위치 근처의 꼭짓점 이름 반환 (없으면 None) """
        for i, (x, y) in enumerate(self.vertex_points):
            dist = math.hypot(pos.x() - x, pos.y() - y)
            if dist < 10:
                return f"Vertex {i + 1}"
        return None

    def find_clicked_anchor(self, pos):
        for anchor_name, widget in self.anchor_labels.items():
            rect = widget.geometry()  # 전체 QLabel 영역
            if rect.contains(pos.toPoint()):
                return anchor_name
        return None

    def confirm_vertex_position(self, x, y):
        reply = QMessageBox.question(
            self.parent, "꼭짓점 위치 수정",
            f"{self.moving_vertex} 위치를 ({x:.2f} m, {y:.2f} m)로 수정할까요?",
            QMessageBox.Yes | QMessageBox.No
        )
        if reply == QMessageBox.Yes:
            self.vertex_data[self.moving_vertex] = {"x": x, "y": y}
            self.edit_workspace()
        self.moving_vertex = None
        self.parent.ui.workspace.setCursor(Qt.ArrowCursor)

    def confirm_anchor_position(self, x, y):
        reply = QMessageBox.question(
            self.parent, "앵커 위치 수정",
            f"{self.moving_anchor} 위치를 ({x:.2f} m, {y:.2f} m)로 수정할까요?",
            QMessageBox.Yes | QMessageBox.No
        )
        if reply == QMessageBox.Yes:
            self.anchor_data[self.moving_anchor]["x"] = x
            self.anchor_data[self.moving_anchor]["y"] = y
            self.edit_workspace()
        self.moving_anchor = None
        self.parent.ui.workspace.setCursor(Qt.ArrowCursor)

    def show_tooltip_during_move(self, event):
        if not hasattr(self, "scale_ratio") or not hasattr(self, "workspace_box"):
            return

        x_real = (event.position().x() - self.workspace_box.x()) / self.scale_ratio
        y_real = (event.position().y() - self.workspace_box.y()) / self.scale_ratio
        tooltip = f"({x_real:.2f} m, {y_real:.2f} m)"

        if hasattr(self, "moving_vertex") and self.moving_vertex:
            QToolTip.showText(event.globalPosition().toPoint(), tooltip, self.parent.ui.workspace)

            # 미리보기 좌표 갱신 (vertex용)
            self.preview_vertex_position = (
                x_real * self.scale_ratio + self.workspace_box.x(),
                y_real * self.scale_ratio + self.workspace_box.y()
            )

            self.parent.ui.workspace.update()

        elif hasattr(self, "moving_anchor") and self.moving_anchor:
            QToolTip.showText(event.globalPosition().toPoint(), tooltip, self.parent.ui.workspace)
            self.parent.ui.workspace.update()

        else:
            QToolTip.hideText()
            self.preview_vertex_position = None


