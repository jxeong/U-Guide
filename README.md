# UWB와 AI를 활용한 휠체어 지하철 승·하차 안전 보조 시스템
GRU 기반 시계열 데이터 처리와 UWB 실시간 위치 추정을 활용한 배리어프리 프로젝트

## 📌 프로젝트 소개
- **목표**: UWB(Ultra-Wideband) 센서를 활용해 휠체어 및 이동 약자의 실내 이동을 보조하는 안전 시스템 개발  
- GRU 시계열 모델을 통한 센서 데이터 분석 + UWB 기반 실시간 위치 추적을 결합  
- 실내 환경에서 정확한 위치 정보와 안전 모니터링 제공


## 📌 데이터셋 다운로드, 폴더 위치
- [YOLO] 강아지 얼굴 - 눈 데이터셋(이미지, 라벨링) -<a href="https://drive.google.com/file/d/1Kpyr5NNtKyTtM7oFbv-JW4uux3fkFRy4/view?usp=sharing" > 구글드라이브 이동</a>
     - YOLO 데이터셋 파일 경로: `ultralytics/custom_train`

- [ResNet] 반려견 백내장 진행 단계 데이터셋(이미지) -<a href="https://drive.google.com/file/d/16yyHc9qtFL8t1XJTO6o2_pUhwn8J6wNV/view?usp=sharing" > 구글드라이브 이동</a>
     - ResNet 데이터셋 파일 경로: `eye/eye_data`


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
```
├── README.md
├── setup.py
├── uguide_data.csv
├── csv_merge.py
├── main.py
├── main.ui
├── icon.ico
│
├── modules
│   ├── __init__.py
│   ├── app_functions.py
│   ├── app_settings.py
│   ├── door_logger.py
│   ├── resources_rc.py
│   ├── serial_handler.py
│   ├── ui_functions.py
│   ├── ui_main.py
│   ├── utils.py
│   ├── uwb_functions.py
│   ├── workspace.db
│   │
│   └── logs
│       └── __init__.py
│
├── widgets
│   └── (커스텀 위젯 관련 코드들)
│
├── csv_files
│   └── (CSV 데이터셋 저장)
│
└── images
    └── (앵커 아이콘, GUI 관련 리소스)
```
