import cv2

def detect_liveness(video_path):
    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()
    
    if not ret:
        print("Unable to read video.")
        return
    
    previous_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    while True:
        ret, frame = cap.read()
        
        if not ret:
            break
        
        current_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame_diff = cv2.absdiff(previous_frame, current_frame)
        
        _, thresholded = cv2.threshold(frame_diff, 25, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresholded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 1000:  # Adjust this threshold based on your setup
                print("Live face detected!")
                return
        
        cv2.imshow("Liveness Detection", frame)
        previous_frame = current_frame
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    video_path = "/home/tenet/Desktop/face_recognition/hi.mp4"  # Replace with the actual video path
    detect_liveness(video_path)
