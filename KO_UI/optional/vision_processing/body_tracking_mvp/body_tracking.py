import math
import time
import cv2
import numpy as np
import mediapipe as mp

try:
    mp_pose = mp.solutions.pose
    mp_drawing = mp.solutions.drawing_utils
except AttributeError:
    import mediapipe.python.solutions.pose as mp_pose
    import mediapipe.python.solutions.drawing_utils as mp_drawing


# =====================================================================
# 1. 속도 및 위치 칼만 필터 (노이즈 억제 및 반응성 균형 조정)
# =====================================================================
class JointKalmanFilter2D:

    def __init__(self, init_pt, process_noise=1e-2, measurement_noise=1e-1):
        self.kf = cv2.KalmanFilter(4, 2)
        self.kf.transitionMatrix = np.array(
            [[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]],
            dtype=np.float32,
        )
        self.kf.measurementMatrix = np.array(
            [[1, 0, 0, 0], [0, 1, 0, 0]], dtype=np.float32
        )
        self.kf.processNoiseCov = np.eye(4, dtype=np.float32) * process_noise
        self.kf.measurementNoiseCov = (
            np.eye(2, dtype=np.float32) * measurement_noise
        )
        self.kf.errorCovPost = np.eye(4, dtype=np.float32)
        self.kf.statePost = np.array(
            [[init_pt[0]], [init_pt[1]], [0], [0]], dtype=np.float32
        )
        self.last_time = time.time()

    def update(self, pt):
        now = time.time()
        dt = now - self.last_time
        self.last_time = now
        if dt <= 0 or dt > 0.1:
            dt = 0.033

        self.kf.transitionMatrix[0, 2] = dt
        self.kf.transitionMatrix[1, 3] = dt

        self.kf.predict()
        measurement = np.array([[pt[0]], [pt[1]]], dtype=np.float32)
        corrected = self.kf.correct(measurement)

        pos = corrected[0:2].flatten()
        vel = corrected[2:4].flatten()
        return pos, vel


def calculate_angle(a, b, c):
    ba, bc = a - b, c - b
    cosine_angle = np.dot(ba, bc) / (
        np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6
    )
    return np.degrees(np.arccos(np.clip(cosine_angle, -1.0, 1.0)))


# =====================================================================
# 2. 메인 예측 파이프라인
# =====================================================================
def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ 웹캠을 열 수 없습니다.")
        return

    pose = mp_pose.Pose(
        min_detection_confidence=0.6,
        min_tracking_confidence=0.6,
        model_complexity=1,
    )

    kf_dict = {}
    velocity_threshold = 25.0  # 감지 문턱값 낮춤
    FULL_EXTENSION_ANGLE = 165.0

    print("=== Improved Impact Prediction System ===")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)  # 거울 모드
        h, w, _ = frame.shape

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(rgb_frame)

        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark

            # 화면상 오른쪽 팔 (사용자 기준 왼쪽 관절)
            shoulder_raw = np.array(
                [
                    landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].x * w,
                    landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].y * h,
                ]
            )
            elbow_raw = np.array(
                [
                    landmarks[mp_pose.PoseLandmark.LEFT_ELBOW].x * w,
                    landmarks[mp_pose.PoseLandmark.LEFT_ELBOW].y * h,
                ]
            )
            wrist_raw = np.array(
                [
                    landmarks[mp_pose.PoseLandmark.LEFT_WRIST].x * w,
                    landmarks[mp_pose.PoseLandmark.LEFT_WRIST].y * h,
                ]
            )

            joints = {
                "shoulder": shoulder_raw,
                "elbow": elbow_raw,
                "wrist": wrist_raw,
            }
            pos_dict, vel_dict = {}, {}

            # 칼만 필터 위치 & 속도 추정
            for name, pt in joints.items():
                if name not in kf_dict:
                    kf_dict[name] = JointKalmanFilter2D(
                        pt, process_noise=1e-2, measurement_noise=1e-1
                    )
                pos_dict[name], vel_dict[name] = kf_dict[name].update(pt)

            elbow_angle = calculate_angle(
                pos_dict["shoulder"], pos_dict["elbow"], pos_dict["wrist"]
            )
            wrist_speed = np.linalg.norm(vel_dict["wrist"])
            is_moving = wrist_speed >= velocity_threshold

            # ★ [핵심 개선] 실제 팔 분할 길이 측정 (상완 + 전완)
            upper_arm_len = np.linalg.norm(
                pos_dict["elbow"] - pos_dict["shoulder"]
            )
            forearm_len = np.linalg.norm(
                pos_dict["wrist"] - pos_dict["elbow"]
            )
            full_arm_reach = upper_arm_len + forearm_len  # 팔 전체 최대 길이

            predicted_impact_pt = None

            # -------------------------------------------------------------
            # [도착 위치 예측 개선 로직]
            # -------------------------------------------------------------
            if is_moving and elbow_angle < FULL_EXTENSION_ANGLE:
                status = "2. PUNCH EXTENDING (Predicting...)"
                color = (0, 255, 255)  # Yellow

                # 진행 방향 벡터 (어깨 -> 손목)
                punch_direction = pos_dict["wrist"] - pos_dict["shoulder"]
                dir_norm = np.linalg.norm(punch_direction)

                if dir_norm > 0:
                    unit_dir = punch_direction / dir_norm

                    # 남은 가숙 확장 거리: (팔 최대 닿는 거리 - 현재 어깨-손목 거리)
                    remaining_reach = max(0, full_arm_reach - dir_norm)

                    # 각도 비율 추가 반영 (팔이 다 펴질 때까지)
                    angle_factor = (
                        FULL_EXTENSION_ANGLE - elbow_angle
                    ) / FULL_EXTENSION_ANGLE
                    predicted_extra_dist = remaining_reach * (
                        1.0 + angle_factor * 0.5
                    )

                    # 손목 기준에서 크게 확장된 최종 예측 지점
                    predicted_impact_pt = (
                        pos_dict["wrist"] + unit_dir * predicted_extra_dist
                    )

            elif elbow_angle >= FULL_EXTENSION_ANGLE - 15 and not is_moving:
                status = "3. FULLY EXTENDED! (Impact Reached)"
                color = (0, 0, 255)  # Red
                predicted_impact_pt = pos_dict["wrist"]
            else:
                status = "1. READY / GUARD"
                color = (255, 191, 0)  # Cyan

            # -------------------------------------------------------------
            # 시각화 (영문 UI 및 조준선)
            # -------------------------------------------------------------
            # 관절 포인트
            for pt in pos_dict.values():
                cv2.circle(frame, (int(pt[0]), int(pt[1])), 6, (0, 255, 0), -1)

            # 팔 스켈레톤
            cv2.line(
                frame,
                (int(pos_dict["shoulder"][0]), int(pos_dict["shoulder"][1])),
                (int(pos_dict["elbow"][0]), int(pos_dict["elbow"][1])),
                color,
                3,
            )
            cv2.line(
                frame,
                (int(pos_dict["elbow"][0]), int(pos_dict["elbow"][1])),
                (int(pos_dict["wrist"][0]), int(pos_dict["wrist"][1])),
                color,
                3,
            )

            # 조준선 시각화
            if predicted_impact_pt is not None:
                target_x, target_y = int(predicted_impact_pt[0]), int(
                    predicted_impact_pt[1]
                )

                # 손목 -> 예상 도착 위치 붉은 가이드라인
                cv2.line(
                    frame,
                    (int(pos_dict["wrist"][0]), int(pos_dict["wrist"][1])),
                    (target_x, target_y),
                    (0, 0, 255),
                    2,
                    cv2.LINE_AA,
                )

                # 조준점 원
                cv2.circle(frame, (target_x, target_y), 18, (0, 0, 255), 2)
                cv2.circle(frame, (target_x, target_y), 4, (0, 0, 255), -1)
                cv2.putText(
                    frame,
                    "PREDICTED IMPACT",
                    (target_x + 15, target_y + 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 0, 255),
                    2,
                )

            # UI 상단 영문 오버레이 (깨짐 방지)
            cv2.rectangle(frame, (10, 10), (520, 85), (0, 0, 0), -1)
            cv2.putText(
                frame,
                f"Status: {status}",
                (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2,
            )
            cv2.putText(
                frame,
                f"Wrist Speed: {int(wrist_speed)} px/s | Angle: {int(elbow_angle)} deg",
                (20, 65),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1,
            )

            mp_drawing.draw_landmarks(
                frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS
            )
        else:
            kf_dict.clear()

        cv2.imshow("Impact Prediction Fixed Demo", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()