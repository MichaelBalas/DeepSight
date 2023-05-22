import argparse

import cv2
import mediapipe as mp
import numpy as np

from helper.iris_lm_depth import from_landmarks_to_depth
from videosource import FileSource, WebcamSource

# Device dictionary for focal length (in px)
device_dict = {
    "phone": 1700,
    "computer": 1000
}

mp_face_mesh = mp.solutions.face_mesh

points_idx = [33, 133, 362, 263]
points_idx = list(set(points_idx))
points_idx.sort()

left_eye_landmarks_id = np.array([33, 133])
right_eye_landmarks_id = np.array([362, 263])

dist_coeff = np.zeros((4, 1))

YELLOW = (0, 255, 255)
GREEN = (0, 255, 0)
BLUE = (255, 0, 0)
LIGHT_RED = (0, 0, 200)
RED = (0, 0, 255)
GREY = (220, 220, 220)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
SMALL_CIRCLE_SIZE = 1
MEDIUM_CIRCLE_SIZE = 2
LARGE_CIRCLE_SIZE = 3


def main(input, device):
    if input is None:
        frame_height, frame_width = (720, 1280)
        source = WebcamSource(width=frame_width, height=frame_height)
    else:
        source = FileSource(input)
        frame_width, frame_height = (int(i) for i in source.get_image_size())

    image_size = (frame_width, frame_height)

    # pseudo camera internals
    focal_length = device_dict.get(device)

    landmarks = None
    smooth_left_depth = -1
    smooth_right_depth = -1
    smooth_factor = 0.1

    with mp_face_mesh.FaceMesh(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5) as face_mesh:
        for idx, (frame, frame_rgb) in enumerate(source):
            results = face_mesh.process(frame_rgb)
            multi_face_landmarks = results.multi_face_landmarks

            if multi_face_landmarks:
                face_landmarks = results.multi_face_landmarks[0]
                landmarks = np.array([(lm.x, lm.y, lm.z) for lm in face_landmarks.landmark])
                landmarks = landmarks.T
                (left_depth, left_iris_size, left_iris_landmarks, left_eye_contours) = from_landmarks_to_depth(frame_rgb, landmarks[:, left_eye_landmarks_id], image_size, is_right_eye=False, focal_length=focal_length)
                (right_depth, right_iris_size, right_iris_landmarks, right_eye_contours) = from_landmarks_to_depth(frame_rgb, landmarks[:, right_eye_landmarks_id], image_size, is_right_eye=True, focal_length=focal_length)

                if smooth_right_depth < 0:
                    smooth_right_depth = right_depth
                else:
                    smooth_right_depth = (smooth_right_depth * (1 - smooth_factor) + right_depth * smooth_factor)

                if smooth_left_depth < 0:
                    smooth_left_depth = left_depth
                else:
                    smooth_left_depth = (smooth_left_depth * (1 - smooth_factor) + left_depth * smooth_factor)

                print(
                    f"depth in cm: {smooth_left_depth / 10:.2f}, {smooth_right_depth / 10:.2f}"
                )
                print(f"size: {left_iris_size:.2f}, {right_iris_size:.2f}")

            if landmarks is not None:

                # draw subset of facemesh
                for ii in points_idx:
                    pos = (np.array(image_size) * landmarks[:2, ii]).astype(np.int32)
                    frame = cv2.circle(frame, tuple(pos), MEDIUM_CIRCLE_SIZE, YELLOW, -1)

                # draw eye contours
                eye_landmarks = np.concatenate(
                    [
                        right_eye_contours,
                        left_eye_contours,
                    ]
                )
                for landmark in eye_landmarks:
                    pos = (np.array(image_size) * landmark[:2]).astype(np.int32)
                    frame = cv2.circle(frame, tuple(pos), SMALL_CIRCLE_SIZE, LIGHT_RED, -1)

                # draw iris landmarks
                iris_landmarks = np.concatenate(
                    [
                        right_iris_landmarks,
                        left_iris_landmarks,
                    ]
                )
                for landmark in iris_landmarks:
                    pos = (np.array(image_size) * landmark[:2]).astype(np.int32)
                    frame = cv2.circle(frame, tuple(pos), LARGE_CIRCLE_SIZE, WHITE, -1)

                # write depth values into frame
                text_color = WHITE
                rectangle_bgr = BLACK
                scale = 1.5
                thickness = 2

                # text and box for OD (right eye)
                text_od = "OD: {:.2f}cm".format(smooth_right_depth / 10)
                (text_width_od, text_height_od) = cv2.getTextSize(text_od, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)[0]
                text_offset_x_od = 20
                text_offset_y_od = text_height_od + 40  # a bit of padding
                box_coords_od = ((text_offset_x_od, text_offset_y_od), (text_offset_x_od + text_width_od + 20, text_offset_y_od - text_height_od - 20))  # more padding
                frame = cv2.rectangle(frame, box_coords_od[0], box_coords_od[1], rectangle_bgr, cv2.FILLED)
                frame = cv2.putText(frame, text_od, (text_offset_x_od + 10, text_offset_y_od - 10), cv2.FONT_HERSHEY_SIMPLEX, scale, text_color, thickness)  # and more padding

                # text and box for OS (left eye)
                text_os = "OS: {:.2f}cm".format(smooth_left_depth / 10)
                (text_width_os, text_height_os) = cv2.getTextSize(text_os, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)[0]
                text_offset_x_os = 20
                text_offset_y_os = 2 * text_height_os + 60  # move down by previous text height and some padding
                box_coords_os = ((text_offset_x_os, text_offset_y_os), (text_offset_x_os + text_width_os + 20, text_offset_y_os - text_height_os - 20))  # more padding
                frame = cv2.rectangle(frame, box_coords_os[0], box_coords_os[1], rectangle_bgr, cv2.FILLED)
                frame = cv2.putText(frame, text_os, (text_offset_x_os + 10, text_offset_y_os - 10), cv2.FONT_HERSHEY_SIMPLEX, scale, text_color, thickness)  # and more padding

            source.show(frame)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Choose video file otherwise webcam is used."
    )
    parser.add_argument(
        "-i", metavar="path-to-file", type=str, help="Path to video file"
    )
    parser.add_argument(
        "-d", metavar="device", type=str, required=True, help="Specify the device type"
    )
    args = parser.parse_args()
    input = args.i; device = args.d
    main(input, device)
