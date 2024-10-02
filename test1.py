import cv2
import dlib
import numpy as np
from scipy.spatial import distance

# 눈의 EAR(Eye Aspect Ratio)를 계산하는 함수
def calculate_ear(eye):
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    C = distance.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

# 눈 깜박임을 감지하는 EAR 임계값 및 프레임 기준 설정
EAR_THRESHOLD = 0.25  # EAR 임계값
EAR_CONSEC_FRAMES = 3  # 연속된 프레임 수
COUNTER = 0  # 깜박임을 세는 변수
TOTAL_BLINKS = 0  # 총 깜박임 수

# 얼굴 인식기 및 랜드마크 예측기 초기화
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# 눈 좌표 인덱스 (dlib 얼굴 랜드마크 68개 기준)
LEFT_EYE_IDX = list(range(36, 42))
RIGHT_EYE_IDX = list(range(42, 48))

# 카메라 장치 초기화
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # 프레임을 그레이스케일로 변환
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 얼굴 탐지
    faces = detector(gray)

    for face in faces:
        # 얼굴 랜드마크 감지
        landmarks = predictor(gray, face)
        landmarks = [(landmarks.part(n).x, landmarks.part(n).y) for n in range(68)]

        # 왼쪽 눈과 오른쪽 눈 좌표 추출
        left_eye = [landmarks[i] for i in LEFT_EYE_IDX]
        right_eye = [landmarks[i] for i in RIGHT_EYE_IDX]

        # EAR 계산
        left_ear = calculate_ear(left_eye)
        right_ear = calculate_ear(right_eye)

        # 양쪽 눈의 평균 EAR 계산
        ear = (left_ear + right_ear) / 2.0

        # EAR 임계값을 기준으로 깜박임 감지
        if ear < EAR_THRESHOLD:
            COUNTER += 1
        else:
            if COUNTER >= EAR_CONSEC_FRAMES:
                TOTAL_BLINKS += 1
                # 깜박임 감지 메시지 표시
                cv2.putText(frame, "Blink!", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 2)
            COUNTER = 0

        # 감지된 눈 주위에 다각형 그리기
        cv2.polylines(frame, [np.array(left_eye, dtype=np.int32)], True, (0, 255, 0), 2)
        cv2.polylines(frame, [np.array(right_eye, dtype=np.int32)], True, (0, 255, 0), 2)

        # EAR 값과 총 깜박임 수를 표시
        cv2.putText(frame, f"EAR: {ear:.2f}", (300, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, f"Blinks: {TOTAL_BLINKS}", (300, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    # 결과 프레임 화면에 표시
    cv2.imshow('Frame', frame)

    # 'q' 키를 누르면 종료
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 카메라 해제 및 창 닫기
cap.release()
cv2.destroyAllWindows()