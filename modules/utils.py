import os
import sys

def resource_path(relative_path):
    """
    cx_Freeze 환경과 개발 환경에서 모두 동작하는 리소스 경로 반환 함수
    """
    if hasattr(sys, '_MEIPASS'):
        # cx_Freeze 실행 파일에서의 리소스 위치
        base_path = os.path.join(sys._MEIPASS, "lib")
    else:
        # 개발 환경의 현재 디렉토리
        base_path = os.path.abspath(".")
    return os.path.join(base_path, relative_path)
