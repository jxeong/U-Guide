import serial
from threading import Thread, Event
from serial.tools import list_ports


def detect_ports():
    ports = list_ports.comports()
    uwb_port = None

    for port in ports:
        desc = port.description.lower()
        if "ch340" in desc:
            uwb_port = port.device

    return uwb_port


class DualSerialHandler:
    def __init__(self, uwb_callback, parent=None):
        self.uwb_serial = None
        self.uwb_callback = uwb_callback
        self.parent = parent

        self.uwb_thread = None
        self.uwb_running = Event()

        self.uwb_port = detect_ports()

    def connect_all(self):
        if self.uwb_port and (not self.uwb_serial or not self.uwb_serial.is_open):
            try:
                print(f"[DEBUG] Connecting UWB at {self.uwb_port}")
                self.uwb_serial = serial.Serial(self.uwb_port, 115200, timeout=1)
                self.uwb_running.set()
                self.uwb_thread = Thread(target=self.read_uwb_data, daemon=True)
                self.uwb_thread.start()
                print(f"[INFO] UWB 포트 연결됨: {self.uwb_port}")
            except Exception as e:
                print(f"[ERROR] UWB 연결 실패: {e}")

    def read_uwb_data(self):
        while self.uwb_running.is_set():
            try:
                line = self.uwb_serial.readline().strip().decode("utf-8", errors="ignore")
                if line:
                    self.uwb_callback(line)
            except Exception as e:
                print(f"[ERROR] UWB 수신 실패: {e}")
                self.uwb_running.clear()


    def disconnect_all(self):
        self.uwb_running.clear()
        if self.uwb_serial and self.uwb_serial.is_open:
            self.uwb_serial.close()

        print("[INFO] 모든 시리얼 포트 연결 종료됨.")
