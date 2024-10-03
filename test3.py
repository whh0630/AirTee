import cv2
import mediapipe as mp

# Mediapipe 초기화
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_drawing = mp.solutions.drawing_utils

# 카메라 장치 초기화
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read() 
    if not ret:
        break

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

    # 결과 프레임을 화면에 표시
    cv2.imshow('Frame', frame)

    # 'q' 키를 누르면 종료
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 카메라 해제 및 창 닫기
cap.release()
cv2.destroyAllWindows()
