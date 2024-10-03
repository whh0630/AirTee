import cv2
import dlib
from scipy.spatial import distance
import numpy as np
import mediapipe as mp

# Mediapipe 초기화
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_drawing = mp.solutions.drawing_utils


# 눈의 EAR(Eye Aspect Ratio)를 계산하는 함수
def calculate_ear(eye):
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    C = distance.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

# 눈의 EAR(Eye Aspect Ratio)를 계산하는 함수
def calculate_ear(eye):
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    C = distance.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

# 입의 벌림 정도를 계산하는 함수
def calculate_mouth_open(mouth):
    A = distance.euclidean(mouth[2], mouth[10])  # 입의 위아래 거리
    B = distance.euclidean(mouth[4], mouth[8])   # 입의 좌우 거리
    C = distance.euclidean(mouth[3], mouth[9])   # 입의 중간 위아래 거리
    #mouth_open_ratio = (A + C) / (2.0 * B)       # 상하 거리와 중간 거리를 평균하여 계산
    mouth_open_ratio = ((A + C + B)/300)       # 상하 거리와 중간 거리를 평균하여 계산
    return [mouth_open_ratio,A,B,C]

def calculate_face_size(landmarks): 
    # 얼굴의 왼쪽 끝 (0번 랜드마크)과 오른쪽 끝 (16번 랜드마크) 사이의 거리 계산
    face_width = distance.euclidean(landmarks[0], landmarks[16])
    return face_width

# 눈 깜박임 및 입 벌림 임계값 설정
EAR_THRESHOLD = 0.19  # 눈 깜박임 임계값
EAR_CONSEC_FRAMES = 3  # 연속된 프레임 수
MOUTH_THRESHOLD = 0.17  # 입 벌림 임계값 
COUNTER = 0  # 깜박임 카운터
TOTAL_BLINKS = 0  # 총 깜박임 수

# 얼굴 인식기 및 랜드마크 예측기 초기화
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# 눈과 입 좌표 인덱스 (dlib 얼굴 랜드마크 68개 기준)
LEFT_EYE_IDX = list(range(36, 42))
RIGHT_EYE_IDX = list(range(42, 48))
MOUTH_IDX = list(range(48, 68))

# 카메라 장치 초기화
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret: 
        break


    # Mediapipe
    # BGR 이미지를 RGB로 변환
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # 신체 랜드마크 감지
    results = pose.process(rgb_frame)

    # 신체 랜드마크가 감지된 경우
    if results.pose_landmarks:
        # 신체 랜드마크 그리기
        mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        # 어깨와 목의 좌표 추출 (좌표는 정규화되어 있으므로 이미지 크기에 맞게 조정 필요)
        image_height, image_width, _ = frame.shape
        left_shoulder = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER]
        right_shoulder = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER]
        nose = results.pose_landmarks.landmark[mp_pose.PoseLandmark.NOSE]

        # 좌표 변환 (정규화된 좌표를 이미지 크기로 변환)
        left_shoulder_point = (int(left_shoulder.x * image_width), int(left_shoulder.y * image_height))
        right_shoulder_point = (int(right_shoulder.x * image_width), int(right_shoulder.y * image_height))
        neck_point = (int(nose.x * image_width), int((left_shoulder.y + right_shoulder.y) / 2 * image_height))

        # 어깨와 목에 원 그리기
        cv2.circle(frame, left_shoulder_point, 5, (0, 255, 0), -1)
        cv2.circle(frame, right_shoulder_point, 5, (0, 255, 0), -1)
        cv2.circle(frame, neck_point, 5, (255, 0, 0), -1)

        # 어깨와 목을 연결하는 선 그리기
        cv2.line(frame, left_shoulder_point, neck_point, (255, 255, 0), 2)
        cv2.line(frame, right_shoulder_point, neck_point, (255, 255, 0), 2)

        dleft_ear = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_EAR]
        dright_ear = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_EAR]
        cv2.putText(frame, "LEFT_EAR: {dleft_ear}", (300, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        #cv2.putText(frame, f"RIGHT_EAR: {dright_ear:.2f}", (300, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

    # opencv dlib
    # 프레임을 그레이스케일로 변환
    gray = rgb_frame #cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # 얼굴 탐지
    faces = detector(gray)
    for face in faces:
        # 얼굴 랜드마크 감지
        landmarks = predictor(gray, face)
        landmarks = [(landmarks.part(n).x, landmarks.part(n).y) for n in range(68)]

        # 왼쪽 눈과 오른쪽 눈 좌표 추출
        left_eye = [landmarks[i] for i in LEFT_EYE_IDX]
        right_eye = [landmarks[i] for i in RIGHT_EYE_IDX]
        mouth = [landmarks[i] for i in MOUTH_IDX]

        # 눈의 EAR 계산
        left_ear = calculate_ear(left_eye)
        right_ear = calculate_ear(right_eye)
        ear = (left_ear + right_ear) / 2.0

        # 입 벌림 정도 계산
        mouth_open_ratio = calculate_mouth_open(mouth)[0]
        mouth_A = calculate_mouth_open(mouth)[1]
        mouth_B = calculate_mouth_open(mouth)[2] 
        mouth_C = calculate_mouth_open(mouth)[3]

        # 얼굴 크기 계산 
        face_size = calculate_face_size(landmarks)

        # 입이 벌어졌는지 감지

        # 눈과 입에 다각형 그리기 
        cv2.polylines(frame, [np.array(left_eye, dtype=np.int32)], True, (0, 255, 0), 1)
        cv2.polylines(frame, [np.array(right_eye, dtype=np.int32)], True, (0, 255, 0), 1)
        cv2.polylines(frame, [np.array(mouth, dtype=np.int32)], True, (255, 255, 0), 1)

        # EAR 값과 총 깜박임 수를 화면에 표시
        # EAR 임계값을 기준으로 깜박임 감지
        if ear < EAR_THRESHOLD:
            COUNTER += 1
            cv2.putText(frame, f"EAR: {ear:.2f}", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        else:
            if COUNTER >= EAR_CONSEC_FRAMES:
                TOTAL_BLINKS += 1
                # 깜박임 감지 메시지 표시 
                cv2.putText(frame, f"EAR: {ear:.2f} Blink", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            else:
                cv2.putText(frame, f"EAR: {ear:.2f}", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            COUNTER = 0
        cv2.putText(frame, f"Blinks: {TOTAL_BLINKS}", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

        # 입 벌림 비율(Mouth Open Ratio)도 화면에 표시
        cv2.putText(frame, f"Mouth Ratio: {(mouth_open_ratio/face_size*100):.2f} / {mouth_A:.2f} / {mouth_B:.2f} / {mouth_C:.2f} / {face_size:.2f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        if (mouth_open_ratio/face_size*100) > MOUTH_THRESHOLD: 
            cv2.putText(frame, "Mouth Open", (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        else:
            cv2.putText(frame, f"Mouth Close", (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)




    # 결과 프레임을 화면에 표시
    cv2.imshow('Frame', frame)

    # 'q' 키를 누르면 종료
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 카메라 해제 및 창 닫기
cap.release()
cv2.destroyAllWindows()
