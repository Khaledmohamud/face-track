import cv2
import mediapipe as mp
import numpy as np
import pyvirtualcam

# Initialize MediaPipe Face Detection and Hands
mp_face_detection = mp.solutions.face_detection
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

face_detection = mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.6)
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5)

# Webcam setup
cap = cv2.VideoCapture(1)

# Stabilization variables
previous_bbox = None
rolling_avg_bbox = None  # Rolling average for smooth tracking
alpha = 0.85  # Smoothing factor for exponential smoothing
rolling_window = 5  # Number of frames for rolling average
zoom_scale = 1.8  # Initial zoom scale

# Threshold to ignore small changes
bbox_threshold = 10

# Rolling average storage
bbox_history = []

def stabilize_bbox(new_bbox, previous_bbox, alpha, threshold):
    """Stabilize the bounding box using exponential smoothing and thresholding."""
    if previous_bbox is None:
        return new_bbox

    # Ignore small changes below the threshold
    if all(abs(new - prev) <= threshold for new, prev in zip(new_bbox, previous_bbox)):
        return previous_bbox

    # Apply exponential smoothing
    return [int(alpha * p + (1 - alpha) * n) for p, n in zip(previous_bbox, new_bbox)]

def rolling_average_bbox(new_bbox):
    """Compute a rolling average for bounding box stabilization."""
    global bbox_history, rolling_window

    bbox_history.append(new_bbox)
    if len(bbox_history) > rolling_window:
        bbox_history.pop(0)

    # Calculate the average bounding box
    avg_bbox = [int(sum(coord) / len(bbox_history)) for coord in zip(*bbox_history)]
    return avg_bbox

def crop_to_face(frame, bbox, scale):
    """Crop and resize the frame around the detected face while maintaining aspect ratio."""
    h, w, _ = frame.shape
    aspect_ratio = w / h  # Calculate the aspect ratio of the output frame
    x_min, y_min, width, height = bbox
    x_center = x_min + width // 2
    y_center = y_min + height // 2

    # Calculate new dimensions while maintaining aspect ratio
    new_width = int(width * scale)
    new_height = int(new_width / aspect_ratio)

    # Ensure the height matches the scaled dimensions
    if new_height > h:
        new_height = h
        new_width = int(new_height * aspect_ratio)

    # Determine cropping bounds
    x_start = max(0, x_center - new_width // 2)
    y_start = max(0, y_center - new_height // 2)
    x_end = min(w, x_start + new_width)
    y_end = min(h, y_start + new_height)

    # Adjust bounds if they exceed frame dimensions
    x_start = max(0, min(x_start, w - new_width))
    y_start = max(0, min(y_start, h - new_height))

    # Crop and resize the region
    cropped = frame[y_start:y_end, x_start:x_end]
    return cv2.resize(cropped, (w, h))

def detect_peace_or_rock(hand_landmarks):
    """Detect peace (✌️) or rock (🤘) gestures."""
    if hand_landmarks:
        # Extract key landmarks
        index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
        index_mcp = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP]
        middle_tip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
        middle_mcp = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_MCP]
        ring_tip = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP]
        ring_mcp = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_MCP]
        pinky_tip = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP]
        pinky_mcp = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_MCP]

        # Peace sign (✌️): Index and middle fingers extended, others folded
        peace = (
            index_tip.y < index_mcp.y and
            middle_tip.y < middle_mcp.y and
            ring_tip.y > ring_mcp.y and
            pinky_tip.y > pinky_mcp.y
        )

        # Rock sign (🤘): Index and pinky fingers extended, others folded
        rock = (
            index_tip.y < index_mcp.y and
            pinky_tip.y < pinky_mcp.y and
            middle_tip.y > middle_mcp.y and
            ring_tip.y > ring_mcp.y
        )

        # Debugging output for gesture detection
        if peace:
            print("Gesture detected: Peace Sign (✌️)")
            return "peace"
        elif rock:
            print("Gesture detected: Rock Sign (🤘)")
            return "rock"
    return None

# Open virtual camera
with pyvirtualcam.Camera(width=640, height=480, fps=30) as virtual_cam:
    print(f"Virtual camera initialized: {virtual_cam.device}")

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        # Flip horizontally for a selfie-view display
        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape

        # Convert BGR to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_results = face_detection.process(rgb_frame)
        hand_results = hands.process(rgb_frame)

        # Gesture detection for both hands
        if hand_results.multi_hand_landmarks:
            detected_gestures = []
            for hand_landmarks in hand_results.multi_hand_landmarks:
                # Detect gestures
                gesture = detect_peace_or_rock(hand_landmarks)
                if gesture:
                    detected_gestures.append(gesture)

            # Act on detected gestures (prioritize "rock" over "peace")
            if "rock" in detected_gestures:
                zoom_scale = min(zoom_scale + 0.1, 3.0)  # Zoom in
            elif "peace" in detected_gestures:
                zoom_scale = max(zoom_scale - 0.1, 1.0)  # Zoom out

        # Face detection
        if face_results.detections:
            # Track the largest face detected (most likely to be the user's)
            largest_face = max(face_results.detections, key=lambda det: det.location_data.relative_bounding_box.width)
            bboxC = largest_face.location_data.relative_bounding_box
            x_min = int(bboxC.xmin * w)
            y_min = int(bboxC.ymin * h)
            width = int(bboxC.width * w)
            height = int(bboxC.height * h)

            # Stabilize bounding box using smoothing and thresholding
            current_bbox = [x_min, y_min, width, height]
            stable_bbox = stabilize_bbox(current_bbox, previous_bbox, alpha, bbox_threshold)

            # Compute rolling average for additional stability
            rolling_avg_bbox = rolling_average_bbox(stable_bbox)

            # Update previous_bbox for next frame
            previous_bbox = stable_bbox

            # Crop and resize around the stabilized face
            frame = crop_to_face(frame, rolling_avg_bbox, zoom_scale)
        else:
            # If no face detected, hold the last known bounding box
            if rolling_avg_bbox:
                frame = crop_to_face(frame, rolling_avg_bbox, zoom_scale)

        # Convert RGB back to BGR for correct color display
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        # Send the processed frame to the virtual camera
        virtual_cam.send(frame)
        virtual_cam.sleep_until_next_frame()

cap.release()