# Face-Recognition-Attendance-with-Liveness-detection

This GitHub repository contains code for a liveness detection and face recognition attendance system. This system combines face recognition technology with liveness detection to mark attendance for individuals. It includes code for capturing images, encoding faces, comparing faces for recognition, and verifying liveness through blinks.

# Prerequisites
Before using this system, ensure you have the following dependencies installed:
- Python 3.x
- OpenCV (cv2)
- imutils
- face_recognition (fr)
- NumPy (numpy)
- dlib
- mediapipe (mediapipe)
- tqdm
- pickle
You can install these dependencies using pip install <package_name>.
# Instructions
1. Clone this repository to your local machine:
2. Download or capture images of individuals whose attendance you want to track. Organize these images into folders, where each folder represents a person's name. Place all these folders in a directory and specify the path to this directory in the code.

3. Encode the faces in the images using the pickle.py script. Replace <path_to_folder_with_folders_of_images> with the path to your image folders and <encodings_pickle_file_name> with the desired name for the pickle file that will store the encodings.
4. Run python pickle.py
5. This script will generate a pickle file containing face encodings for known individuals.

6. Update the main script attendance_system.py with the path to your encodings pickle file and the folder containing images of individuals for whom you want to mark attendance.

7. Run the attendance system using the following command:
    python attendance_system.py
   
The system will capture video from your default camera and recognize individuals. It will then perform liveness detection by asking a random question and tracking eye blinks. If liveness is successfully detected, attendance will be marked for that individual.
# Configuration
In the attendance_system.py script, you can configure several parameters such as:

- limit_questions: The number of liveness detection questions asked per person.
- limit_consecutives: The number of consecutive successful liveness detections required.
- limit_try: The number of attempts allowed for each liveness detection question.
- delay_between_detections: The delay in seconds between detecting individuals.
You can adjust these parameters according to your requirements.

# Acknowledgments
This project uses various libraries and techniques for face recognition and liveness detection. Thanks to the developers of these libraries for their contributions.
I would like to acknowledge the following open-source project for their contributions:
- [juan-csv/face_liveness_detection-Anti-spoofing](https://github.com/juan-csv/face_liveness_detection-Anti-spoofing): This repository provided valuable insights and code that influenced this project.



