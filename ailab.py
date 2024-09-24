from flask import Flask, Response, render_template
import pygame
import os
import mediapipe as mp
import cv2
import time
import random

app = Flask(__name__)

@app.route("/")
def index():
    return render_template('index.html')

@app.route("/game.html")
def game():
    return render_template('game.html')

def run_game():
    pygame.init()
    os.environ["OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS"] = "0" 

    sound_path = "D:\\2023expo\\2023-expo\static\\sound\\"
    image_path = "D:\\2023expo\\2023-expo\static\image\\"
    video_path = "D:\\2023expo\\2023-expo\static\\video\\"

    rand = random.randint(1,4)

    sound = pygame.mixer.Sound(sound_path + "sound.wav")
    minussound = pygame.mixer.Sound(sound_path + "minussound.wav")

    userlist = ['user{}'.format(i) for i in range(1,11)] # 사용자 이름 저장
    result = [random.randint(0,70) for i in range(10)] # 사용자의 점수를 차례로 저장
    user = 1  # 사용자 이름 뒤에 붙일 숫자

    username = f'user{user}' # 사용자의 이름 

    max_score = []

    for i in range(1,4):
        if rand == i:
            pygame.mixer.music.load(sound_path + f"Music{str(rand)}.mp3")
            plus3 = cv2.resize(cv2.imread(image_path + f'plus3{str(rand)}.png', cv2.IMREAD_UNCHANGED),(200,200))
            plus5 = cv2.resize(cv2.imread(image_path + f'plus5{str(rand)}.png', cv2.IMREAD_UNCHANGED),(100,100))
            minus = cv2.resize(cv2.imread(image_path + f'minus{str(rand)}.png', cv2.IMREAD_UNCHANGED),(400,400))


    # if rand == 0:
    #     pygame.mixer.music.load("Music1.mp3")
    #     plus3 = cv2.imread('plus31.png', cv2.IMREAD_UNCHANGED)
    #     plus3 = cv2.resize(plus3, (200,200))
    #     plus5 = cv2.imread('plus51.png', cv2.IMREAD_UNCHANGED)
    #     plus5 = cv2.resize(plus5, (100,100))
    #     minus = cv2.imread('minus1.png', cv2.IMREAD_UNCHANGED)
    #     minus = cv2.resize(minus, (400,400))

    # elif rand == 1:
    #     pygame.mixer.music.load("Music2.mp3") # 해 
    #     plus3 = cv2.imread('plus32.png', cv2.IMREAD_UNCHANGED)
    #     plus3 = cv2.resize(plus3, (200,200))
    #     plus5 = cv2.imread('plus52.png', cv2.IMREAD_UNCHANGED)
    #     plus5 = cv2.resize(plus5, (100,100))
    #     minus = cv2.imread('minus2.png', cv2.IMREAD_UNCHANGED)
    #     minus = cv2.resize(minus, (400,400))

        
    # elif rand == 2:
    #     pygame.mixer.music.load("Music3.mp3") # 공 
    #     plus3 = cv2.imread('plus33.png', cv2.IMREAD_UNCHANGED)
    #     plus3 = cv2.resize(plus3, (200,200))
    #     plus5 = cv2.imread('plus53.png', cv2.IMREAD_UNCHANGED)
    #     plus5 = cv2.resize(plus5, (100,100))
    #     minus = cv2.imread('minus3.png', cv2.IMREAD_UNCHANGED)
    #     minus = cv2.resize(minus, (400,400))

    # else:
    #     pygame.mixer.music.load("Music4.mp3") # 우주
    #     plus3 = cv2.imread('plus34.png', cv2.IMREAD_UNCHANGED)
    #     plus3 = cv2.resize(plus3, (200,200))
    #     plus5 = cv2.imread('plus54.png', cv2.IMREAD_UNCHANGED)
    #     plus5 = cv2.resize(plus5, (100,100))
    #     minus = cv2.imread('minus4.png', cv2.IMREAD_UNCHANGED)
    #     minus = cv2.resize(minus, (400,400))

    mp_drawing = mp.solutions.drawing_utils
    mp_hands = mp.solutions.hands
    score=0

    game_start_event = False
    game_over_event = False

    time_given=20.9
    time_remaining = 99
    
    x1_enemy=random.randint(100,1180)
    y1_enemy=random.randint(100,620)

    x2_enemy=random.randint(50,1230)
    y2_enemy=random.randint(50,650)

    x3_enemy=random.randint(400,880)
    y3_enemy=random.randint(400,500)

    x4_enemy=random.randint(100,1180)
    y4_enemy=random.randint(100,620)

    effect_small = cv2.resize(cv2.imread(image_path+'effectplus.png', cv2.IMREAD_UNCHANGED),(100,100))
    effect_big = cv2.resize(cv2.imread(image_path+'effectplus.png', cv2.IMREAD_UNCHANGED),(200,200))
    effect_large = cv2.resize(cv2.imread(image_path+'effectminus.png', cv2.IMREAD_UNCHANGED),(400,400))


    # 배경 이미지

    background1 = cv2.resize(cv2.imread(image_path + 'background1.jpg', cv2.IMREAD_UNCHANGED),(1280,720))
    background2 = cv2.resize(cv2.imread(image_path + 'background2.jpg', cv2.IMREAD_UNCHANGED),(1280,720))
    background3 = cv2.resize(cv2.imread(image_path + 'background3.jpg', cv2.IMREAD_UNCHANGED),(1280,720))
    background4 = cv2.resize(cv2.imread(image_path + 'background4.png', cv2.IMREAD_UNCHANGED),(1280,720))
    ready_image = cv2.resize(cv2.imread(image_path + 'ready.png', cv2.IMREAD_UNCHANGED),(1280,720))
    black_image = cv2.resize(cv2.imread(image_path + 'black.png', cv2.IMREAD_UNCHANGED),(1280,720))

    # 이동 속도 설정
    speed = 1 # 추가

    def overlay(image, x, y, w, h, overlay_image): # 대상 이미지 (3채널), x, y 좌표, width, height, 덮어씌울 이미지 (4채널)
        alpha = overlay_image[:, :, 3] # BGRA
        mask_image = alpha / 255 # 0 ~ 255 -> 255 로 나누면 0 ~ 1 사이의 값 (1: 불투명, 0: 완전)
        
        for c in range(0, 3): # channel BGR
            image[y-h:y+h, x-w:x+w, c] = (overlay_image[:, :, c] * mask_image) + (image[y-h:y+h, x-w:x+w, c] * (1 - mask_image))

    # 영상의 프레임 속도를 60 FPS로 설정
    frame_rate = 60

    cap = cv2.VideoCapture(video_path + 'gamestart.mp4')

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

        if cv2.waitKey(1000 // frame_rate) == 27:
            break


        img = cv2.resize(cv2.imread(image_path + 'background1.jpg'),(1280,720))

        img = cv2.rectangle(img, (0,0), (1280, 720), (0,0,0), -1)

        cv2.putText(img, 'Welcome USER{}!'.format(user),
                                (1280//2-400, 1280//2-300),
                                cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 255, 255), 10, cv2.LINE_AA)
        

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

            image = cv2.addWeighted(image,0,background3,1,0) # 우주배경 입히기 
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

                                    if ( 423 < int(pixelCoordinatesLandmark[0]) < 523 and 423 < int(pixelCoordinatesLandmark[1]) < 523):
                                        cap = cv2.VideoCapture(video_path + 'loading.mp4')
                                        while True:
                                            ret, frame = cap.read()
                                            if not ret:
                                                print('cannot read capture image')
                                                break

                                            reg, buffer = cv2.imencode('.jpg', frame)
                                            a = buffer.tobytes()  # 인코딩된 이미지를 바이트 스트림으로 변환합니다.
                                            #   multipart/x-mixed-replace 포맷으로 비디오 프레임을 클라이언트에게 반환합니다.
                                            yield (b'--frame\r\n'
                                                    b'Content-Type: image/jpeg\r\n\r\n' + a + b'\r\n')
                                            

                                        game_start_event = True
                                        start_time = time.time()
                                        pygame.mixer.music.play()
                                        

                                if game_start_event == True and time_remaining > 0:

                                    if rand == 0:
                                        image = cv2.addWeighted(image,0,background1,1,0)
                                        
                                    elif rand == 1:
                                        image = cv2.addWeighted(image,0,background2,1,0)
                                                
                                    elif rand == 2:
                                        image = cv2.addWeighted(image,0,background3,1,0)
                                    
                                    else:
                                        image = cv2.addWeighted(image,0,background4,1,0)

                                    if results.multi_hand_landmarks: # 손 랜드마크 표시
                                        for num, hand in enumerate(results.multi_hand_landmarks):
                                            mp_drawing.draw_landmarks(image, hand, mp_hands.HAND_CONNECTIONS, 
                                                                    mp_drawing.DrawingSpec(color=(255,255,144), thickness=2, circle_radius=2),)

                                    time_remaining = int(time_given - (present_time - start_time))

                                    # 이미지 이동
                                    x1_enemy -= speed
                                    x2_enemy += speed
                                    y3_enemy -= speed

                                    if x1_enemy-50 < int(pixelCoordinatesLandmark[0])< x1_enemy+50 and y1_enemy-50 < int(pixelCoordinatesLandmark[1]) < y1_enemy+50 :
                                        overlay(image, x1_enemy, y1_enemy, 100, 100, effect_big)
                                        sound.play()
                                        x1_enemy=random.randint(100,1180)
                                        y1_enemy=random.randint(100,620)
                                        x2_enemy=random.randint(100,1180)
                                        y2_enemy=random.randint(100,620)

                                        x3_enemy=random.randint(400,880)
                                        y3_enemy=random.randint(400,500)

                                        x4_enemy=random.randint(100,1180)
                                        y4_enemy=random.randint(100,620)

                                        score=score+3
                                        font=cv2.FONT_HERSHEY_SIMPLEX
                                        color=(255,0,255)
                                        text=cv2.putText(frame,"Score",(100,100),font,1,color,4,cv2.LINE_AA)
                                    
                                    if x2_enemy-50 < int(pixelCoordinatesLandmark[0])< x2_enemy+50 and y2_enemy-50 < int(pixelCoordinatesLandmark[1]) < y2_enemy+50 :
                                        overlay(image, x2_enemy, y2_enemy, 50, 50, effect_small)
                                        sound.play()
                                        x1_enemy=random.randint(100,1180)
                                        y1_enemy=random.randint(100,620)
                                        x2_enemy=random.randint(100,1180)
                                        y2_enemy=random.randint(100,620)

                                        x3_enemy=random.randint(400,880)
                                        y3_enemy=random.randint(400,500)

                                        x4_enemy=random.randint(100,1180)
                                        y4_enemy=random.randint(100,620)
                                        score=score+5

                                        font=cv2.FONT_HERSHEY_SIMPLEX
                                        color=(255,0,255)
                                        text=cv2.putText(frame,"Score",(100,100),font,1,color,4,cv2.LINE_AA)
                                        
                                    if x3_enemy-100 < int(pixelCoordinatesLandmark[0])< x3_enemy+100 and y3_enemy-100 < int(pixelCoordinatesLandmark[1]) < y3_enemy+100 :
                                        overlay(image, x3_enemy, y3_enemy, 200, 200, effect_large)
                                        minussound.play()
                                        x1_enemy=random.randint(100,1180)
                                        y1_enemy=random.randint(100,620)

                                        x2_enemy=random.randint(50,1230)
                                        y2_enemy=random.randint(50,650)

                                        x3_enemy=random.randint(400,880)
                                        y3_enemy=random.randint(400,500)

                                        x4_enemy=random.randint(100,1180)
                                        y4_enemy=random.randint(100,620)
                                        
                                        score=score-10
                                        font=cv2.FONT_HERSHEY_SIMPLEX
                                        color=(255,0,255)
                                        text=cv2.putText(frame,"Score",(100,100),font,1,color,4,cv2.LINE_AA)

                                    if x4_enemy-50 < int(pixelCoordinatesLandmark[0])< x4_enemy+50 and y4_enemy-50 < int(pixelCoordinatesLandmark[1]) < y4_enemy+50 :
                                        overlay(image, x4_enemy, y4_enemy, 100, 100, effect_big)
                                        sound.play()
                                        x1_enemy=random.randint(100,1180)
                                        y1_enemy=random.randint(100,620)

                                        x2_enemy=random.randint(50,1230)
                                        y2_enemy=random.randint(50,650)

                                        x3_enemy=random.randint(400,880)
                                        y3_enemy=random.randint(400,500)

                                        x4_enemy=random.randint(100,1180)
                                        y4_enemy=random.randint(100,620)
                                        
                                        score=score+20

                                        font=cv2.FONT_HERSHEY_SIMPLEX
                                        color=(255,0,255)
                                        text=cv2.putText(frame,"Score",(100,100),font,1,color,4,cv2.LINE_AA)
                                    

                                    if x1_enemy >= 1100:
                                        x1_enemy += speed

                                    elif x1_enemy <= 100:
                                        x1_enemy += speed
                                        
                                    if x2_enemy >= 1150:
                                        x2_enemy -= speed
                                    elif x2_enemy <= 50:
                                        x2_enemy -= speed
                                    
                                    if y3_enemy >= 490:
                                        y3_enemy += speed
                                    elif y3_enemy <= 100:
                                        y3_enemy += speed

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
                                            
                                elif game_start_event == True and time_remaining == 0:
                                    time_remaining = 0
                                    game_over_event = True
                                    pygame.mixer.music.stop()
                                    max_score.append(score)
                                    userlist = userlist + [username]
                                    result = result + [score]
                                
                                if game_over_event == True:

                                    image = cv2.rectangle(image, (0,0), (1280, 720), (0,0,0), -1)

                                    cv2.putText(image, 'Game Over',
                                        (w//2-147, h//2-65),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1.7, (255, 255, 255), 3, cv2.LINE_AA)
                                        
                                    cv2.putText(image, 'Your Score:',
                                                                (w//2-120, h//2),
                                                                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA) 

                                    cv2.putText(image, str(score),
                                                                (w//2+80, h//2+3),
                                                                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 2, cv2.LINE_AA)
                                    
                                    cv2.putText(image, 'Good For You!!',
                                                                (w//2-200, h//2+70),
                                                                cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 228, 255), 3, cv2.LINE_AA) 

                            except:
                                    pass
                            
            else:
                image = cv2.imread(image_path + 'handout.png',cv2.IMREAD_GRAYSCALE)

            reg, buffer = cv2.imencode('.jpg', image)
            a = buffer.tobytes()  # 인코딩된 이미지를 바이트 스트림으로 변환합니다.
            #   multipart/x-mixed-replace 포맷으로 비디오 프레임을 클라이언트에게 반환합니다.
            yield (b'--frame\r\n'
                    b'Content-Type: image/jpeg\r\n\r\n' + a + b'\r\n')
            

@app.route('/game_result')
def game_result():
    # 게임 결과를 표시하는 템플릿을 렌더링
    # return render_template('index.html')
    return Response(run_game(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)