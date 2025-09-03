# UWB와 AI를 활용한 휠체어 지하철 승·하차 안전 보조 시스템
GRU 기반 시계열 데이터 처리와 UWB 실시간 위치 추정을 활용한 배리어프리 프로젝트

## 📌 프로젝트 소개
- **목표**: UWB(Ultra-Wideband) 센서를 활용해 휠체어 및 이동 약자의 실내 이동을 보조하는 안전 시스템 개발  
- GRU 시계열 모델을 통한 센서 데이터 분석 + UWB 기반 실시간 위치 추적을 결합  
- 실내 환경에서 정확한 위치 정보와 안전 모니터링 제공

## 💡 실행 방법
### 환경 설정
- Python 3.10 이상
- 주요 라이브러리: `PySide6`, `pandas`, `numpy`, `torch`, `cx_Freeze`

### GUI 실행
```bash
python main.py
```

### ⚙️ 개발 환경
- GPU: `NVIDIA GeForce RTX 3060 Ti`
- `Pycharm` 가상환경


## 📁 프로젝트 구조
```bash
U-Guide/
├── artifacts/                # 빌드/학습 산출물
│   ├── Intent_gru.pt         # 학습된 GRU 모델 가중치
│   └── scale.json            # 입력값 평균·표준편차 및 판별 기준 임계값
├── board_code/               # 보드용 펌웨어 코드(ESP32S3 UWB AT Demo 등)
├── images/                   # GUI 아이콘 및 리소스 이미지
├── modules/                  # 애플리케이션 핵심 모듈
│   ├── app_functions.py      # 주요 로직 함수 정의
│   ├── app_settings.py       # 설정값 및 전역 상수
│   ├── door_logger.py        # 딥러닝 모델 학습 데이터셋 수집
│   ├── serial_handler.py     # UWB-보드 ↔ PC 직렬통신 처리
│   ├── uwb_functions.py      # UWB 데이터 처리 및 좌표 측위 함수
│   ├── ui_functions.py       # UI 이벤트 처리
│   ├── ui_main.py            # 메인 UI 레이아웃
│   ├── utils.py              # 공용 유틸 함수
│   ├── resources_rc.py       # Qt 리소스 바인딩 파일
│   ├── workspace.db          # 워크스페이스 DB (환경 저장)
│   └── logs/                 # 실행 로그 저장
├── widgets/                  # 커스텀 위젯 모음
│
├── main.py                   # GUI 실행 진입점 (PySide6)
├── main.ui                   # Qt Designer UI 파일
├── setup.py                  # cx_Freeze 패키징 스크립트
├── intent_model_train.py      # 승·하차 의도 인식 모델 학습 코드
├── csv_merge.py              # CSV 통합/전처리 스크립트
├── uguide_data.csv           # 기본/샘플 데이터셋
├── resources.qrc             # Qt 리소스 정의 파일
├── icon.ico                  # 애플리케이션 아이콘
├── README.md                 # 프로젝트 설명 문서
└── .gitignore                # Git 관리 제외 설정