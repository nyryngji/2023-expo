import cv2
import random

# 화면 크기 설정
screen_width = 1280
screen_height = 720

# OpenCV 초기화
cap = cv2.VideoCapture(0)  # 비디오 캡처 기본 카메라 사용
cap.set(3, screen_width)  # 화면 너비 설정
cap.set(4, screen_height)  # 화면 높이 설정

# 이미지 로드
image_left = cv2.imread("moon.png")
image_left = cv2.resize(image_left, (200,200))
image_right = cv2.imread("moon.png")
image_right = cv2.resize(image_right, (200,200))
image_up = cv2.imread("moon.png")
image_up = cv2.resize(image_up, (200,200))

# 이미지 초기 위치 설정
x_left, y_left = random.randint(0, screen_width - image_left.shape[1]), random.randint(0, screen_height - image_left.shape[0])
x_right, y_right = random.randint(0, screen_width - image_right.shape[1]), random.randint(0, screen_height - image_right.shape[0])
x_up, y_up = random.randint(0, screen_width - image_up.shape[1]), random.randint(0, screen_height - image_up.shape[0])

# 이동 속도 설정
speed = 1
while True:
    ret, frame = cap.read()

    # 이미지 이동
    x_left -= speed
    x_right += speed
    y_up -= speed

    # 이미지가 화면 밖으로 나가면 초기 위치로 설정
    if x_left + image_left.shape[1] < 0:
        x_left = screen_width
    if x_right > screen_width:
        x_right = 0 - image_right.shape[1]
    if y_up + image_up.shape[0] < 0:
        y_up = screen_height

    # 이미지를 화면에 그립니다.
    frame[y_left:y_left+image_left.shape[0], x_left:x_left+image_left.shape[1]] = image_left
    frame[y_right:y_right+image_right.shape[0], x_right:x_right+image_right.shape[1]] = image_right
    frame[y_up:y_up+image_up.shape[0], x_up:x_up+image_up.shape[1]] = image_up

    cv2.imshow('Game', frame)

    if cv2.waitKey(1) == 27:
        break

cv2.destroyAllWindows()
cap.release()
