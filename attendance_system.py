import random 
import cv2
import imutils
import f_liveness_detection
import questions
import face_recognition as fr
import numpy as np
import os
import pickle
from datetime import datetime
import csv
import time
import mediapipe as mp

# parameters
COUNTER, TOTAL = 0,0
counter_ok_questions = 0
counter_ok_consecutives = 0
limit_consecutives = 2
limit_questions = 3
counter_try = 0
limit_try = 20
delay_between_detections = 3  # Delay in seconds

path_to_faces = '/path to folder with images'
names = []
myList = os.listdir(path_to_faces)
images = []
fr_counter = 0
# cap = cv2.VideoCapture(0)
cam = cv2.VideoCapture(0)
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils
face_detection = mp_face_detection.FaceDetection( model_selection=0, min_detection_confidence=0.7)
# results = face_detection.process(im)


for persons in sorted(os.listdir(path_to_faces)):
    # print(persons)
    for image in sorted(os.listdir(os.path.join(path_to_faces, persons))):
        # print(image)
        names.append(os.path.splitext(image)[0])


def create_new_attendance_file(attendance_file):
    with open(attendance_file, 'w', newline='') as f:
        fieldnames = ['Name', 'Check-in', 'Check-out']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

def mark_attendance(name_from_face_recognition, attendance_file):
    now = datetime.now().strftime('%H:%M:%S')
    checkout_time = now
    records = []
    if os.path.exists(attendance_file):
        with open(attendance_file, 'r') as f:
            reader = csv.DictReader(f)
            records = list(reader)

    for record in records:
        if record['Name'] == name_from_face_recognition:
            record['Check-out'] = checkout_time
            break
    else:
        records.append({
            'Name': name_from_face_recognition,
            'Check-in': now,
            'Check-out': None
        })

    with open(attendance_file, 'w', newline='') as f:
        fieldnames = ['Name', 'Check-in', 'Check-out']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)
    print(f"Attendence of {name_from_face_recognition} marked successfully")    
    return "successful"

def show_image(cam,text,color = (0,0,255)):
    ret, im = cam.read()
    im = imutils.resize(im, width=720)
    im = cv2.flip(im, 1)
    face_cur_frame = fr.face_locations(im)
    cv2.putText(im,text,(10,50),cv2.FONT_HERSHEY_COMPLEX,1,color,2)
    results = face_detection.process(im)
    if results.detections:
        for detection in results.detections:
            for faceLoc in face_cur_frame:
                mp_drawing.draw_detection(im, detection)
                y1,x2,y2,x1 = faceLoc
                y1,x2,y2,x1 = y1*2,x2*2,y2*2,x1*2
                cv2.putText(im,text,(x1-6,y2-6),cv2.FONT_HERSHEY_COMPLEX,1,(255,255,255),)          
        
    return im


def predict_liveness(name_from_face_recognition,attendance_file):
    for i_questions in range(0,limit_questions):
        index_question = random.randint(0,3)
        question = questions.question_bank(index_question)
        img = show_image(cam,question)
        
        for i_try in range(limit_try):
            
            ret, img = cam.read()
            img = imutils.resize(img, width=720)
            img = cv2.flip(img, 1)
            global TOTAL
            global COUNTER
            global counter_try
            global counter_ok_consecutives
            global counter_ok_questions
            TOTAL_0 = TOTAL
            out_model = f_liveness_detection.detect_liveness(img,COUNTER,TOTAL_0)
            TOTAL = out_model['total_blinks']
            COUNTER = out_model['count_blinks_consecutives']
            dif_blink = TOTAL-TOTAL_0
            if dif_blink > 0:
                blinks_up = 1
            else:
                blinks_up = 0

            challenge_res = questions.challenge_result(question, out_model,blinks_up)
            print(challenge_res)

            img = show_image(cam,question)
            cv2.imshow(f'liveness_detection for {name_from_face_recognition}',img)
            if cv2.waitKey(1) &0xFF == ord('q'):
                break 

            if challenge_res == "pass":
                print("four")
                img = show_image(cam,question+" : ok")
                cv2.imshow(f'liveness_detection for {name_from_face_recognition}',img)
                if cv2.waitKey(1) &0xFF == ord('q'):
                    break

                counter_ok_consecutives += 1
                if counter_ok_consecutives == limit_consecutives:
                    counter_ok_questions += 1
                    counter_try = 0
                    counter_ok_consecutives = 0
                    break
                else:
                    continue

            elif challenge_res == "fail":
                counter_try += 1
                show_image(cam,question+" : fail")
            elif i_try == limit_try-1:
                break

               

        if counter_ok_questions ==  limit_questions:
            while True:
                img = show_image(cam,f"LIVENESS SUCCESSFUL for {name_from_face_recognition}" ,color = (0,255,0))
             
                x = mark_attendance(name_from_face_recognition, attendance_file)
               
                cv2.imshow(f'liveness_detection for {name_from_face_recognition}',img)

                if cv2.waitKey(1) &0xFF == ord('q'):
                    break

                if x == "successful":
                    print("attendence marked successfully")
                    # cam.release()
                    # cv2.destroyAllWindows()
                    time.sleep(5)
                    cv2.destroyAllWindows()

                    start()

                    
                else:
                    pass

        elif i_try == limit_try-1:
            while True:
                img = show_image(cam,"LIVENESS FAIL")
                print("Try Again")
                cv2.imshow(f'liveness_detection for {name_from_face_recognition}',img)
                if cv2.waitKey(1) &0xFF == ord('q'):
                    break
                cv2.destroyAllWindows()
                start()

                
            break

        else:
            pass
          
def main():
    while True:
        ret, img_main = cam.read()
        # if ret:
        imgS = cv2.resize(img_main, (0, 0), None, 0.5, 0.5)
        imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)
        face_cur_frame = fr.face_locations(imgS)
        encodes_cur_frame = fr.face_encodings(imgS, face_cur_frame)

        for encodeFace, faceLoc in zip(encodes_cur_frame, face_cur_frame):
            with open('name of encoding file', 'rb') as f:
                encodeListKnown = pickle.load(f)
            matches = fr.compare_faces(encodeListKnown, encodeFace, tolerance=0.42)
            faceDis = fr.face_distance(encodeListKnown, encodeFace)

            if matches[np.argmin(faceDis)]:
                best_match_index = np.argmin(faceDis)
                name_from_face_recognition = names[best_match_index]
                print(name_from_face_recognition)
                
                return name_from_face_recognition 
    
def start():
    x = main()
    attendance_file = f'attendance_{datetime.now().strftime("%d-%m-%Y")}.csv'
    if not os.path.exists(attendance_file):
        create_new_attendance_file(attendance_file)
    
    predict_liveness(x,attendance_file)

if __name__ == '__main__':
    # x = main()
    # attendance_file = f'attendance_{datetime.now().strftime("%d-%m-%Y")}.csv'
    # if not os.path.exists(attendance_file):
    #     create_new_attendance_file(attendance_file)
    
    # predict_liveness(x,attendance_file)
    start()


    # cam.release()
    # cv2.destroyAllWindows()

