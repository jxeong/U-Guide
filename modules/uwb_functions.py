# calculation.py

import math

import numpy as np
from filterpy.kalman import UnscentedKalmanFilter as UKF
from filterpy.kalman import MerweScaledSigmaPoints

# 상태 전이 함수: 거리가 시간에 따라 변하지 않는다고 가정
def fx(x, dt):
    return x

# 측정 함수: 상태값 그대로가 측정값이라고 가정 (직접 관측)
def hx(x):
    return x

# 개별 입자 정의: 각 입자에는 3개의 UKF가 있으며, 각각 range1, range2, range3에 대응
class Particle:
    def __init__(self, init_value=100.0):
        self.ukfs = []
        for _ in range(3):  # 앵커가 3개이므로 UKF도 3개 생성
            points = MerweScaledSigmaPoints(n=1, alpha=0.1, beta=2., kappa=0)  # 시그마 포인트 설정
            ukf = UKF(dim_x=1, dim_z=1, fx=fx, hx=hx, dt=1.0, points=points)   # 1차원 UKF 구성
            ukf.x = np.array([init_value])    # 초기 추정값 (거리)
            ukf.P *= 100.0                    # 초기 오차 공분산
            ukf.Q *= 1.0                      # 프로세스 노이즈 공분산
            ukf.R *= 10.0                     # 측정 노이즈 공분산
            self.ukfs.append(ukf)             # 각 UKF를 리스트에 저장

    def predict(self):
        # 모든 UKF에 대해 예측 단계 수행
        for ukf in self.ukfs:
            ukf.predict()

    def update(self, measurements):
        # 측정값(거리)으로 각 UKF 업데이트
        for i in range(3):
            self.ukfs[i].update(np.array([measurements[i]]))

    def get_filtered_ranges(self):
        # 현재 입자의 UKF들이 추정한 거리값들 반환
        return [ukf.x[0] for ukf in self.ukfs]

# 입자 필터 정의: 여러 입자를 관리하고 평균을 통해 최종 거리 추정값 생성
class ParticleFilter:
    def __init__(self, num_particles=35, init_value=100.0):
        # 지정된 수의 입자 생성 (기본값: 35개)
        self.particles = [Particle(init_value) for _ in range(num_particles)]

    def step(self, measurement):
        # 각 입자에 대해 예측 및 업데이트 수행
        for p in self.particles:
            p.predict()
            p.update(measurement)

    def get_estimates(self):
        # 모든 입자의 추정 거리값을 평균 내어 최종 필터 결과 반환
        all_filtered = np.array([p.get_filtered_ranges() for p in self.particles])
        return np.mean(all_filtered, axis=0)

class Calculation:
    def __init__(self, anchor_count, anchor_offsets = None):
        """
        :param anchor_count: 동적으로 설정된 앵커 수
        """

        self.anchor_offsets = anchor_offsets
        '''
        self.kalman_filters = {
            f"Anchor {i}": KalmanFilter(process_variance=0.1, measurement_variance=0.1)
            for i in range(anchor_count)
        }'''
        self.kalman_filters = {}
        # print(f"Kalman Filters Initialized: {list(self.kalman_filters.keys())}")

    def apply_correction_and_particle_single(self, raw_range, tag_id, anchor_index):
        """
        단일 앵커에 대해 거리 보정 및 UKF 필터 적용
        """
        key = f"Tag_{tag_id}_Anchor_{anchor_index}"
        if key not in self.kalman_filters:
            # 1차원 UKF 하나만 갖는 Particle 생성
            points = MerweScaledSigmaPoints(n=1, alpha=0.1, beta=2.0, kappa=0)
            ukf = UKF(dim_x=1, dim_z=1, fx=fx, hx=hx, dt=1.0, points=points)
            ukf.x = np.array([100.0])
            ukf.P *= 100.0
            ukf.Q *= 1.0
            ukf.R *= 10.0
            self.kalman_filters[key] = ukf

        if raw_range is None:
            return 0

        # 보정
        if raw_range > 300:
            offset = 90
        else:
            offset = 45
        corrected = (raw_range - offset) * 0.01

        ukf = self.kalman_filters[key]
        ukf.predict()
        ukf.update(np.array([corrected]))

        return ukf.x[0]

    def circle_intersections(self, c1, r1, c2, r2, max_adjustments=5):
        """
        두 원의 교점을 계산하는 함수.
        원이 교차하지 않거나 내접 상태일 경우 보정 반지름을 적용하여 교점을 찾음.
        """
        x1, y1 = c1
        x2, y2 = c2
        d = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)  # 두 중심 간의 거리

        adjustments = 0  # 반지름 보정 횟수 제한

        # 교차하지 않는 경우: 반지름 합보다 중심 거리가 크다면 반지름을 확장하여 해결
        while d > r1 + r2 and adjustments < max_adjustments:
            expansion = (d - (r1 + r2)) / 2
            r1 += expansion
            r2 += expansion
            adjustments += 1
            # print(f"[INFO] 반지름 확장: r1={r1}, r2={r2}, 조정 횟수={adjustments}")

        # 내접 상태: 반지름 차보다 중심 거리가 작다면 반지름을 축소하여 해결
        while d < abs(r1 - r2) and adjustments < max_adjustments:
            contraction = (abs(r1 - r2) - d) / 2
            r1 -= contraction
            r2 -= contraction
            adjustments += 1
            # print(f"[INFO] 반지름 축소: r1={r1}, r2={r2}, 조정 횟수={adjustments}")

        # 보정 횟수가 초과되면 종료
        if adjustments >= max_adjustments:
            # print("[WARNING] 최대 반지름 보정 횟수를 초과하여 교점 계산을 중단합니다.")
            return None

        # 중심이 동일하면 교점 계산 불가
        if d == 0:
            # print("[WARNING] 두 원의 중심이 동일하여 교점을 찾을 수 없습니다.")
            return None

        # 두 원이 교차한다고 간주하여 교점 계산
        a = (r1 ** 2 - r2 ** 2 + d ** 2) / (2 * d)
        h = math.sqrt(max(r1 ** 2 - a ** 2, 0))  # max로 음수 방지

        x3 = x1 + a * (x2 - x1) / d
        y3 = y1 + a * (y2 - y1) / d

        offset_x = h * (y2 - y1) / d
        offset_y = h * (x2 - x1) / d

        return (x3 + offset_x, y3 - offset_y), (x3 - offset_x, y3 + offset_y)

    def distance(self, point1, point2):
        """
        두 점 사이의 유클리드 거리를 계산합니다.

        Parameters:
            point1: 첫 번째 점 (x1, y1) 형태의 튜플
            point2: 두 번째 점 (x2, y2) 형태의 튜플

        Returns:
            두 점 사이의 거리 (float)
        """
        x1, y1 = point1
        x2, y2 = point2
        return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

    def closest_point(self, points, target):
        return min(points, key=lambda p: self.distance(p, target))

    def generalized_trilateration(self, valid_anchors, all_anchors):
        """
        Generalized trilateration using only valid anchors for intersection,
        but evaluates errors using all anchors (including those outside range bounds).

        Parameters:
            valid_anchors: list of anchor dicts with keys 'index', 'range', 'position'
            all_anchors: list of all anchors to evaluate distance error

        Returns:
            (x, y): Refined position as the average of valid intersection points.
        """
        points = []
        checked_pairs = set()
        num_valid = len(valid_anchors)

        for i in range(num_valid):
            for j in range(i + 1, num_valid):
                a_i = valid_anchors[i]
                a_j = valid_anchors[j]

                if (a_i["index"], a_j["index"]) in checked_pairs:
                    continue

                #print(f"[교점 계산] 앵커 {a_i['index']} ↔ 앵커 {a_j['index']}")

                intersections = self.circle_intersections(
                    a_i["position"], a_i["range"],
                    a_j["position"], a_j["range"]
                )

                if intersections:
                    # 교점 후보 중 거리 오차가 가장 적은 점 선택 (all_anchors 기준)
                    #print(f' inter: {intersections}')
                    closest_intersection = min(
                        intersections,
                        key=lambda point: sum(
                            abs(self.distance(point, a["position"]) - a["range"])
                            for a in all_anchors if a["index"] not in (a_i["index"], a_j["index"])
                        )
                    )

                    #print(f"  ↳ 선택된 교점: {closest_intersection}")
                    points.append(closest_intersection)

                checked_pairs.add((a_i["index"], a_j["index"]))

        if points:
            x = sum(p[0] for p in points) / len(points)
            y = sum(p[1] for p in points) / len(points)
            #print(f"[결과] 평균 교점 위치: ({round(x, 2)}, {round(y, 2)})")
            return round(x, 2), round(y, 2)

        print("[경고] 유효한 교점 없음 → 위치 계산 실패")
        return None, None
