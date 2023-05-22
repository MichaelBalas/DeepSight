import argparse
import cv2
import mediapipe as mp
import numpy as np
import os

from helper.iris_lm_depth import from_landmarks_to_depth
from videosource import FileSource

# Device dictionary for focal length (in px)
device_dict = {
    "phone": 1700,
    "computer": 1000
}

mp_face_mesh = mp.solutions.face_mesh

left_eye_landmarks_id = np.array([33, 133])
right_eye_landmarks_id = np.array([362, 263])


def displayIrisDiameter(left_iris, right_iris):
    print(f"Average iris sizes in mm: {left_iris:.2f}, {right_iris:.2f}")
    average_iris_size = (left_iris + right_iris) / 2
    print(f"Average iris size in mm: {average_iris_size:.2f}")
    
def main(input, device, distance):
    total_left_iris_size = 0
    total_right_iris_size = 0
    count_frames = 0
    # Identify if the input is a video or image by the file extension
    file_extension = os.path.splitext(input)[1]
    if file_extension in ['.jpg', '.jpeg', '.png']:
        source = FileSource(input)
        frame_width, frame_height = (int(i) for i in source.get_image_size())
        image_size = (frame_width, frame_height)
        focal_length = device_dict.get(device)*2
        landmarks = None
        smooth_left_size = -1
        smooth_right_size = -1
        smooth_factor = 0.1
        
        with mp_face_mesh.FaceMesh(static_image_mode=True, min_detection_confidence=0.5, min_tracking_confidence=0.5) as face_mesh:
            for idx, (frame, frame_rgb) in enumerate(source):
                results = face_mesh.process(frame_rgb)
                multi_face_landmarks = results.multi_face_landmarks
                if multi_face_landmarks:
                    face_landmarks = results.multi_face_landmarks[0]
                    landmarks = np.array([(lm.x, lm.y, lm.z) for lm in face_landmarks.landmark])
                    landmarks = landmarks.T
                    (left_depth, left_iris_size, left_iris_landmarks, left_eye_contours) = from_landmarks_to_depth(frame_rgb, landmarks[:, left_eye_landmarks_id], image_size, is_right_eye=False, focal_length=focal_length)
                    (right_depth, right_iris_size, right_iris_landmarks, right_eye_contours) = from_landmarks_to_depth(frame_rgb, landmarks[:, right_eye_landmarks_id], image_size, is_right_eye=True, focal_length=focal_length)
                    # Calculate real iris sizes (in mm) using camera pinhole model
                    left_iris_real_size = left_iris_size * distance / focal_length
                    right_iris_real_size = right_iris_size * distance / focal_length
                    if smooth_right_size < 0:
                        average_right_iris_size = right_iris_real_size
                    else:
                        average_right_iris_size = (smooth_right_size * (1 - smooth_factor)) + (right_iris_real_size * smooth_factor)
                    if smooth_left_size < 0:
                        average_left_iris_size = left_iris_real_size
                    else:
                        average_left_iris_size = (smooth_left_size * (1 - smooth_factor)) + (left_iris_real_size * smooth_factor)
                    displayIrisDiameter(average_right_iris_size, average_left_iris_size)
                    
            source.release()
            cv2.destroyAllWindows()
            
    elif file_extension in ['.mp4', '.avi']:
        video = cv2.VideoCapture(input)
        frame_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        image_size = (frame_width, frame_height)
        focal_length = device_dict.get(device)

        with mp_face_mesh.FaceMesh(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5) as face_mesh:
            while video.isOpened():
                ret, frame = video.read()
                if not ret:
                    break
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = face_mesh.process(frame_rgb)
                multi_face_landmarks = results.multi_face_landmarks
                if multi_face_landmarks:
                    face_landmarks = results.multi_face_landmarks[0]
                    landmarks = np.array([(lm.x, lm.y, lm.z) for lm in face_landmarks.landmark])
                    landmarks = landmarks.T
                    (left_depth, left_iris_size, left_iris_landmarks, left_eye_contours) = from_landmarks_to_depth(frame_rgb, landmarks[:, left_eye_landmarks_id], image_size, is_right_eye=False, focal_length=focal_length)
                    (right_depth, right_iris_size, right_iris_landmarks, right_eye_contours) = from_landmarks_to_depth(frame_rgb, landmarks[:, right_eye_landmarks_id], image_size, is_right_eye=True, focal_length=focal_length)
                    # Calculate real iris sizes (in mm) using camera pinhole model
                    left_iris_real_size = left_iris_size * distance / focal_length
                    right_iris_real_size = right_iris_size * distance / focal_length
                    # Accumulate total iris sizes and increment frame count
                    total_left_iris_size += left_iris_real_size
                    total_right_iris_size += right_iris_real_size
                    count_frames += 1
            # Calculate average iris sizes
            average_left_iris_size = total_left_iris_size / count_frames
            average_right_iris_size = total_right_iris_size / count_frames
            displayIrisDiameter(average_right_iris_size, average_left_iris_size)
                
            video.release()
    else:
        print("Unsupported file type. Please use image or video files only.")
        return


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", type=str, help="Path to the image or video file", required=True)
    parser.add_argument("-d", "--device", type=str, help="Device used for capturing the image or video", required=True)
    parser.add_argument("-z", "--distance", default=30, type=float, help="Distance to the face in cm", required=True)
    args = parser.parse_args()
    main(args.input, args.device, args.distance * 10)  # Convert cm to mm
