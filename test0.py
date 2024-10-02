import cv2

# 배경 차감 알고리즘 생성
back_sub = cv2.createBackgroundSubtractorMOG2()

# 카메라 장치 초기화
cap = cv2.VideoCapture(0)

while True:
    # 카메라로부터 프레임 읽기
    ret, frame = cap.read()

    if not ret:
        break

    # 배경 차감 적용하여 움직임을 감지
    fg_mask = back_sub.apply(frame)

    # 움직임이 감지된 부분을 강조하기 위해 외곽선을 찾아서 그리기
    contours, _ = cv2.findContours(fg_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        if cv2.contourArea(contour) > 500:  # 너무 작은 움직임은 무시
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # 결과를 화면에 표시
    cv2.imshow('Frame', frame)
    cv2.imshow('FG Mask', fg_mask)

    # 'q' 키를 누르면 종료
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 모든 창을 닫고 카메라 해제
cap.release()
cv2.destroyAllWindows()