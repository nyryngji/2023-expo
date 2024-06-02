from flask import Flask, Response, render_template, send_from_directory, redirect, url_for
import pygame
import os
import mediapipe as mp
import cv2
import numpy as np
from mediapipe.framework.formats import landmark_pb2
import time
import random
import sys

app = Flask(__name__)

@app.route("/")
def index():
    return render_template('index.html')

@app.route("/game.html")
def game():
    return render_template('game.html')


# 이미지 파일을 제공하기 위한 라우트
@app.route('/image/')
def get_image(filename):
    return send_from_directory('static/image', filename)

# 동영상 파일을 제공하기 위한 라우트
@app.route('/video/')
def get_video(filename):
    return send_from_directory('static/video', filename)

# @app.route('/run_game')
def run_game():
  
    pygame.init()
    os.environ["OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS"] = "0" 

    rand = random.randint(0,3)

    if rand == 0:
        pygame.mixer.music.load("Don_t Let Me Down.mp3") # 육 
        plus3 = cv2.imread('plus3gro.png', cv2.IMREAD_UNCHANGED)
        plus3 = cv2.resize(plus3, (200,200))
        plus5 = cv2.imread('plus3gro.png', cv2.IMREAD_UNCHANGED)
        plus5 = cv2.resize(plus5, (100,100))
        minus = cv2.imread('minusgro.png', cv2.IMREAD_UNCHANGED)
        minus = cv2.resize(minus, (400,400))
    elif rand == 1:
        pygame.mixer.music.load("Tropic Love.mp3") # 해 
        plus3 = cv2.imread('plus3sea.png', cv2.IMREAD_UNCHANGED)
        plus3 = cv2.resize(plus3, (200,200))
        plus5 = cv2.imread('plus5sea.png', cv2.IMREAD_UNCHANGED)
        plus5 = cv2.resize(plus5, (100,100))
        minus = cv2.imread('minussea.png', cv2.IMREAD_UNCHANGED)
        minus = cv2.resize(minus, (400,400))
    elif rand == 2:
        pygame.mixer.music.load("Stockholm Lights.mp3") # 공 
        plus3 = cv2.imread('plus3sky.png', cv2.IMREAD_UNCHANGED)
        plus3 = cv2.resize(plus3, (200,200))
        plus5 = cv2.imread('plus5sky.png', cv2.IMREAD_UNCHANGED)
        plus5 = cv2.resize(plus5, (100,100))
        minus = cv2.imread('minussky.png', cv2.IMREAD_UNCHANGED)
        minus = cv2.resize(minus, (400,400))
    else:
        pygame.mixer.music.load("Stay.mp3") # 우주
        plus3 = cv2.imread('plus3gal.png', cv2.IMREAD_UNCHANGED)
        plus3 = cv2.resize(plus3, (200,200))
        plus5 = cv2.imread('plus5gal.png', cv2.IMREAD_UNCHANGED)
        plus5 = cv2.resize(plus5, (100,100))
        minus = cv2.imread('minusgal.png', cv2.IMREAD_UNCHANGED)
        minus = cv2.resize(minus, (400,400))

    mp_drawing = mp.solutions.drawing_utils
    mp_hands = mp.solutions.hands
    score=0

    game_start_event = False
    game_over_event = False
    game_pause_event = False

    time_given=40.9
    time_remaining = 99
    
    x1_enemy=random.randint(100,1180)
    y1_enemy=random.randint(100,620)

    x2_enemy=random.randint(50,1230)
    y2_enemy=random.randint(50,650)

    x3_enemy=random.randint(400,880)
    y3_enemy=random.randint(400,500)

    effect_small = cv2.imread('effect.png', cv2.IMREAD_UNCHANGED)
    effect_small = cv2.resize(effect_small, (100,100))

    effect_big = cv2.imread('effect.png', cv2.IMREAD_UNCHANGED)
    effect_big = cv2.resize(effect_big, (200,200))

    effect_large = cv2.imread('minuseffect.png', cv2.IMREAD_UNCHANGED)
    effect_large = cv2.resize(effect_large, (400,400))

    # 배경 이미지

    galaxy_image = cv2.imread('galaxy.jpg', cv2.IMREAD_UNCHANGED)
    galaxy_image = cv2.resize(galaxy_image, (1280,720))

    ready_image = cv2.imread('ready.png', cv2.IMREAD_UNCHANGED)
    ready_image = cv2.resize(ready_image, (1280,720))

    galaxy_image = cv2.imread('galaxy.jpg', cv2.IMREAD_UNCHANGED)
    galaxy_image = cv2.resize(galaxy_image, (1280,720))

    ground_image = cv2.imread('ground.jpg', cv2.IMREAD_UNCHANGED)
    ground_image = cv2.resize(ground_image, (1280,720))

    sea_image = cv2.imread('sea.png', cv2.IMREAD_UNCHANGED)
    sea_image = cv2.resize(sea_image, (1280,720))

    sky_image = cv2.imread('sky.jpg', cv2.IMREAD_UNCHANGED)
    sky_image = cv2.resize(sky_image, (1280,720))

    def overlay(image, x, y, w, h, overlay_image): # 대상 이미지 (3채널), x, y 좌표, width, height, 덮어씌울 이미지 (4채널)
        alpha = overlay_image[:, :, 3] # BGRA
        mask_image = alpha / 255 # 0 ~ 255 -> 255 로 나누면 0 ~ 1 사이의 값 (1: 불투명, 0: 완전)
        
        for c in range(0, 3): # channel BGR
            image[y-h:y+h, x-w:x+w, c] = (overlay_image[:, :, c] * mask_image) + (image[y-h:y+h, x-w:x+w, c] * (1 - mask_image))

    cap = cv2.VideoCapture('gamestart.mp4')

    while True:
        ret, frame = cap.read()
        if not ret:
            print('cannot read capture image')
            break

        # cv2.imshow('show',cv2.resize(frame,(1280,720)))
        reg, buffer = cv2.imencode('.jpg', frame)
        a = buffer.tobytes()  # 인코딩된 이미지를 바이트 스트림으로 변환합니다.
            #   multipart/x-mixed-replace 포맷으로 비디오 프레임을 클라이언트에게 반환합니다.
        yield (b'--frame\r\n'
                    b'Content-Type: image/jpeg\r\n\r\n' + a + b'\r\n')

        if cv2.waitKey(1) == 27:
            break

    video = cv2.VideoCapture(0) # 640 x 480

    with mp_hands.Hands(min_detection_confidence=0.8, min_tracking_confidence=0.5) as hands: 
        while video.isOpened():
            _, frame = video.read()            
    
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            image = cv2.flip(image, 1)
            image = cv2.resize(image,(1280,720))
            
            h, w, _ = image.shape

            present_time = time.time()
    
            results = hands.process(image)
    
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            image2 = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            image = cv2.addWeighted(image,0,galaxy_image,1,0) # 우주배경 입히기 
            image = cv2.addWeighted(image,0,ready_image,1,0) # 우주 배경 입힌거 위에 ready 이미지

            if results.multi_hand_landmarks: # 손 랜드마크 표시
                for num, hand in enumerate(results.multi_hand_landmarks):
                    mp_drawing.draw_landmarks(image, hand, mp_hands.HAND_CONNECTIONS, 
                                            mp_drawing.DrawingSpec(color=(255,255,144), thickness=2, circle_radius=2),)
            

            if results.multi_hand_landmarks != None:
                for handLandmarks in results.multi_hand_landmarks:
                    for point in mp_hands.HandLandmark:
                        normalizedLandmark = handLandmarks.landmark[point]
                        pixelCoordinatesLandmark = mp_drawing._normalized_to_pixel_coordinates(normalizedLandmark.x, normalizedLandmark.y, w, h)

                        if point==8:
                            try:
                                if game_start_event == False:
                                    # cv2.circle(image,(int(pixelCoordinatesLandmark[0]),int(pixelCoordinatesLandmark[1])),20,(255,0,0),-1)

                                    if ( 423 < int(pixelCoordinatesLandmark[0]) < 523 and 423 < int(pixelCoordinatesLandmark[1]) < 523):
                                        cap = cv2.VideoCapture('loading.mp4')
                                        while True:
                                            ret, frame = cap.read()
                                            if not ret:
                                                print('cannot read capture image')
                                                break

                                            # cv2.imshow('show',cv2.resize(frame,(1280,720)))
                                            reg, buffer = cv2.imencode('.jpg', frame)
                                            a = buffer.tobytes()  # 인코딩된 이미지를 바이트 스트림으로 변환합니다.
                                            #   multipart/x-mixed-replace 포맷으로 비디오 프레임을 클라이언트에게 반환합니다.
                                            yield (b'--frame\r\n'
                                                    b'Content-Type: image/jpeg\r\n\r\n' + a + b'\r\n')
                                            

                                            # if cv2.waitKey(1) == 27:
                                            #     break

                                        game_start_event = True
                                        start_time = time.time()
                                        pygame.mixer.music.play()
                                        

                                if game_start_event == True and time_remaining > 0:

                                    if rand == 0:
                                        image = cv2.addWeighted(image2,0,ground_image,1,0)
                                        
                                    elif rand == 1:
                                        image = cv2.addWeighted(image2,0,sea_image,1,0)
                                                
                                    elif rand == 2:
                                        image = cv2.addWeighted(image2,0,sky_image,1,0)
                                    
                                    else:
                                        image = cv2.addWeighted(image2,0,galaxy_image,1,0)

                                    if results.multi_hand_landmarks: # 손 랜드마크 표시
                                        for num, hand in enumerate(results.multi_hand_landmarks):
                                            mp_drawing.draw_landmarks(image, hand, mp_hands.HAND_CONNECTIONS, 
                                                                    mp_drawing.DrawingSpec(color=(255,255,144), thickness=2, circle_radius=2),)

                                    time_remaining = int(time_given - (present_time - start_time))

                                    if x1_enemy-50 < int(pixelCoordinatesLandmark[0])< x1_enemy+50 and y1_enemy-50 < int(pixelCoordinatesLandmark[1]) < y1_enemy+50 :
                                        overlay(image, x1_enemy, y1_enemy, 100, 100, effect_big)
                                        x1_enemy=random.randint(100,1180)
                                        y1_enemy=random.randint(100,620)
                                        x2_enemy=random.randint(100,1180)
                                        y2_enemy=random.randint(100,620)

                                        x3_enemy=random.randint(400,880)
                                        y3_enemy=random.randint(400,500)
                                        score=score+5
                                        font=cv2.FONT_HERSHEY_SIMPLEX
                                        color=(255,0,255)
                                        text=cv2.putText(frame,"Score",(100,100),font,1,color,4,cv2.LINE_AA)
                                    
                                    if x2_enemy-50 < int(pixelCoordinatesLandmark[0])< x2_enemy+50 and y2_enemy-50 < int(pixelCoordinatesLandmark[1]) < y2_enemy+50 :
                                        overlay(image, x2_enemy, y2_enemy, 50, 50, effect_small)
                                        x1_enemy=random.randint(100,1180)
                                        y1_enemy=random.randint(100,620)
                                        x2_enemy=random.randint(100,1180)
                                        y2_enemy=random.randint(100,620)

                                        x3_enemy=random.randint(400,880)
                                        y3_enemy=random.randint(400,500)
                                        score=score+1
                                        font=cv2.FONT_HERSHEY_SIMPLEX
                                        color=(255,0,255)
                                        text=cv2.putText(frame,"Score",(100,100),font,1,color,4,cv2.LINE_AA)
                                        
                                    if x3_enemy-100 < int(pixelCoordinatesLandmark[0])< x3_enemy+100 and y3_enemy-100 < int(pixelCoordinatesLandmark[1]) < y3_enemy+100 :
                                        overlay(image, x3_enemy, y3_enemy, 200, 200, effect_large)
                                        x1_enemy=random.randint(100,1180)
                                        y1_enemy=random.randint(100,620)

                                        x2_enemy=random.randint(50,1230)
                                        y2_enemy=random.randint(50,650)

                                        x3_enemy=random.randint(400,880)
                                        y3_enemy=random.randint(400,500)
                                        score=score-10
                                        font=cv2.FONT_HERSHEY_SIMPLEX
                                        color=(255,0,255)
                                        text=cv2.putText(frame,"Score",(100,100),font,1,color,4,cv2.LINE_AA)
                                    

                                    cv2.putText(image, 'Score:',
                                            (w//2-250, 35),
                                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255,255,255), 2, cv2.LINE_AA)      

                                    cv2.putText(image, str(score),
                                            (w//2-130, 35),
                                                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255,255,255), 2, cv2.LINE_AA)         

                                    cv2.putText(image, 'Time left:',
                                            (w//2+30, 35),
                                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255,255,255), 2, cv2.LINE_AA)      

                                    cv2.putText(image, str(time_remaining),
                                            (w//2+230, 35),
                                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255,255,255), 2, cv2.LINE_AA) 
                                    
                                    overlay(image, x1_enemy, y1_enemy, 100, 100, plus3)
                                    overlay(image, x2_enemy, y2_enemy, 50, 50, plus5)
                                    overlay(image, x3_enemy, y3_enemy, 200, 200, minus)

                                elif game_start_event == True and time_remaining <= 0:
                                    time_remaining = 0
                                    game_over_event = True
                                
                                if game_over_event == True:

                                    image = cv2.addWeighted(image,0,galaxy_image,1,0)
                        
                                    cv2.rectangle(image, (w//2-170, h//2-130), (w//2+170, h//2+40), (0,0,0), -1)

                                    cv2.putText(image, 'Game Over',
                                    (w//2-147, h//2-65),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1.7, (255, 255, 255), 3, cv2.LINE_AA)
                                    
                                    cv2.putText(image, 'Your Score:',
                                    (w//2-120, h//2),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA) 

                                    cv2.putText(image, str(score),
                                    (w//2+80, h//2+3),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 2, cv2.LINE_AA)

                                    pygame.mixer.music.stop()

                            except:
                                    pass
            
            # cv2.imshow('show', image)
            reg, buffer = cv2.imencode('.jpg', image)
            a = buffer.tobytes()  # 인코딩된 이미지를 바이트 스트림으로 변환합니다.
            #   multipart/x-mixed-replace 포맷으로 비디오 프레임을 클라이언트에게 반환합니다.
            yield (b'--frame\r\n'
                    b'Content-Type: image/jpeg\r\n\r\n' + a + b'\r\n')
            
            # if cv2.waitKey(1) == 27:
            #     pygame.mixer.music.stop()
            #     break
                

    # video.release()
    # cv2.destroyAllWindows()
    # return redirect(url_for('game_result'))  # 게임 종료 후 game_result 페이지로 리디렉션

@app.route('/game_result')
def game_result():
    # 게임 결과를 표시하는 템플릿을 렌더링
    # return render_template('index.html')
    return Response(run_game(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)