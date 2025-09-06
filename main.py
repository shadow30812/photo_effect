import math
import time

import cv2
import mediapipe as mp
import numpy as np

# --- Initialization ---
# Initialize MediaPipe Hands, Face Mesh and drawing utilities
mp_hands = mp.solutions.hands
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils

# Create a Hands instance for hand tracking
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5,
)

# Create a FaceMesh instance for face landmark detection
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
)

# Start video capture from the default webcam
cap = cv2.VideoCapture(0)


# --- Filter & Effect Functions ---


def filter_bw(frame, **kwargs):
    """Converts the frame to black and white."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)


def filter_invert(frame, **kwargs):
    """Inverts the colors of the frame."""
    return cv2.bitwise_not(frame)


def filter_thermal(frame, **kwargs):
    """Applies a thermal-like colormap to the frame."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return cv2.applyColorMap(gray, cv2.COLORMAP_JET)


def filter_depth(frame, **kwargs):
    """Applies a depth-map-like colormap to the frame."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return cv2.applyColorMap(gray, cv2.COLORMAP_BONE)


def effect_dog_ears(image, face_landmarks, w, h):
    """Draws dog ears on the detected face."""
    outer_ear = np.array(
        [[0, 0], [40, -80], [80, 0], [60, 40], [20, 40]], dtype=np.int32
    )
    inner_ear = np.array([[20, 25], [40, -40], [60, 25]], dtype=np.int32)

    left_temple = face_landmarks.landmark[162]
    right_temple = face_landmarks.landmark[389]

    lx, ly = int(left_temple.x * w), int(left_temple.y * h)
    rx, ry = int(right_temple.x * w), int(right_temple.y * h)

    head_width = np.hypot(lx - rx, ly - ry)
    scale = head_width / 250.0

    angle_rad = math.atan2(ry - ly, rx - lx)
    angle_deg = math.degrees(angle_rad)

    # Left Ear
    M_left = cv2.getRotationMatrix2D((0, 0), angle_deg, scale)
    anchor_left = (lx - int(20 * scale), ly - int(100 * scale))
    transformed_outer_left = (
        cv2.transform(outer_ear.reshape(-1, 1, 2), M_left).reshape(-1, 2) + anchor_left
    )
    transformed_inner_left = (
        cv2.transform(inner_ear.reshape(-1, 1, 2), M_left).reshape(-1, 2) + anchor_left
    )
    cv2.fillPoly(image, [transformed_outer_left.astype(np.int32)], (80, 50, 45))
    cv2.fillPoly(image, [transformed_inner_left.astype(np.int32)], (220, 160, 200))

    # Right Ear
    outer_ear_right, inner_ear_right = (
        outer_ear * np.array([[-1, 1]]),
        inner_ear * np.array([[-1, 1]]),
    )
    M_right = cv2.getRotationMatrix2D((0, 0), angle_deg, scale)
    anchor_right = (rx + int(20 * scale), ry - int(100 * scale))
    transformed_outer_right = (
        cv2.transform(outer_ear_right.reshape(-1, 1, 2), M_right).reshape(-1, 2)
        + anchor_right
    )
    transformed_inner_right = (
        cv2.transform(inner_ear_right.reshape(-1, 1, 2), M_right).reshape(-1, 2)
        + anchor_right
    )
    cv2.fillPoly(image, [transformed_outer_right.astype(np.int32)], (80, 50, 45))
    cv2.fillPoly(image, [transformed_inner_right.astype(np.int32)], (220, 160, 200))
    return image


def effect_moustache(image, face_landmarks, w, h):
    """Draws a moustache on the detected face."""
    upper_lip = face_landmarks.landmark[13]
    mouth_left = face_landmarks.landmark[61]
    mouth_right = face_landmarks.landmark[291]

    ux, uy = int(upper_lip.x * w), int(upper_lip.y * h)
    lx, ly = int(mouth_left.x * w), int(mouth_left.y * h)
    rx, ry = int(mouth_right.x * w), int(mouth_right.y * h)

    scale = np.hypot(lx - rx, ly - ry) / 120.0
    angle_rad = math.atan2(ry - ly, rx - lx)
    angle_deg = math.degrees(angle_rad)

    moustache_shape = np.array(
        [[-50, 0], [-40, -10], [0, -5], [40, -10], [50, 0], [0, 10]], dtype=np.int32
    )
    M = cv2.getRotationMatrix2D((0, 0), angle_deg, scale)
    anchor = (ux, uy + int(10 * scale))
    transformed_moustache = (
        cv2.transform(moustache_shape.reshape(-1, 1, 2), M).reshape(-1, 2) + anchor
    )
    cv2.fillPoly(image, [transformed_moustache.astype(np.int32)], (40, 40, 40))
    return image


def effect_goofy_eyes(image, face_landmarks, w, h):
    """Draws large cartoon eyes over the user's eyes."""
    left_eye_center = np.mean(
        [
            (lm.x, lm.y)
            for i, lm in enumerate(face_landmarks.landmark)
            if i in [33, 160, 158, 133, 153, 144]
        ],
        axis=0,
    )
    right_eye_center = np.mean(
        [
            (lm.x, lm.y)
            for i, lm in enumerate(face_landmarks.landmark)
            if i in [362, 385, 387, 263, 373, 380]
        ],
        axis=0,
    )

    eye_width = np.hypot(
        face_landmarks.landmark[33].x - face_landmarks.landmark[133].x,
        face_landmarks.landmark[33].y - face_landmarks.landmark[133].y,
    )
    radius = int(eye_width * w * 0.8)

    lex, ley = int(left_eye_center[0] * w), int(left_eye_center[1] * h)
    rex, rey = int(right_eye_center[0] * w), int(right_eye_center[1] * h)

    cv2.circle(image, (lex, ley), radius, (255, 255, 255), -1)
    cv2.circle(image, (rex, rey), radius, (255, 255, 255), -1)
    cv2.circle(image, (lex + int(radius / 4), ley), int(radius / 2), (0, 0, 0), -1)
    cv2.circle(image, (rex - int(radius / 4), rey), int(radius / 2), (0, 0, 0), -1)
    return image


def effect_pixelate_face(image, face_landmarks, w, h):
    """Pixelates the region of the detected face."""
    points = np.array(
        [(int(lm.x * w), int(lm.y * h)) for lm in face_landmarks.landmark]
    )
    x, y, w_box, h_box = cv2.boundingRect(points)

    # Add a little padding
    x, y = max(0, x - 10), max(0, y - 10)
    w_box, h_box = w_box + 20, h_box + 20

    face_roi = image[y : y + h_box, x : x + w_box]
    if face_roi.size == 0:
        return image

    # Pixelate
    small = cv2.resize(face_roi, (20, 20), interpolation=cv2.INTER_LINEAR)
    pixelated = cv2.resize(small, (w_box, h_box), interpolation=cv2.INTER_NEAREST)

    image[y : y + h_box, x : x + w_box] = pixelated
    return image


# --- Main Application State ---
# Categorize effects for easier logic
ROI_FILTERS = [filter_bw, filter_invert, filter_thermal, filter_depth]
FACE_EFFECTS = [
    effect_dog_ears,
    effect_goofy_eyes,
    effect_pixelate_face,
]

# Combine all functions and names for cycling
all_effects = ROI_FILTERS + FACE_EFFECTS
effect_names = [
    "Black & White",
    "Invert",
    "Thermal",
    "Depth",
    "Dog Ears",
    "Goofy Eyes",
    "Pixelate Face",
]
current_effect_index = 0

last_pinch_time = 0
PINCH_DEBOUNCE_TIME = 0.5
PINCH_THRESHOLD = 30
WINDOW_NAME = "Hand Gesture Filter"

try:
    # --- Main Loop ---
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        results_hands = hands.process(rgb_frame)
        results_faces = face_mesh.process(rgb_frame)
        output_frame = frame.copy()

        # Apply face-based effects if one is selected and a face is detected
        active_effect = all_effects[current_effect_index]
        if active_effect in FACE_EFFECTS and results_faces.multi_face_landmarks:
            for face_landmarks in results_faces.multi_face_landmarks:
                output_frame = active_effect(output_frame, face_landmarks, w, h)

        # Hand tracking and gesture control logic
        if results_hands.multi_hand_landmarks:
            left_pinch, right_pinch = False, False
            left_hand_points, right_hand_points = None, None

            for hand_landmarks, handedness in zip(
                results_hands.multi_hand_landmarks, results_hands.multi_handedness
            ):
                thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
                index_tip = hand_landmarks.landmark[
                    mp_hands.HandLandmark.INDEX_FINGER_TIP
                ]
                x_thumb, y_thumb = int(thumb_tip.x * w), int(thumb_tip.y * h)
                x_index, y_index = int(index_tip.x * w), int(index_tip.y * h)

                if np.hypot(x_thumb - x_index, y_thumb - y_index) < PINCH_THRESHOLD:
                    if handedness.classification[0].label == "Left":
                        left_pinch = True
                    else:
                        right_pinch = True

                if handedness.classification[0].label == "Left":
                    left_hand_points = [(x_thumb, y_thumb), (x_index, y_index)]
                else:
                    right_hand_points = [(x_thumb, y_thumb), (x_index, y_index)]

                mp_drawing.draw_landmarks(
                    frame, hand_landmarks, mp_hands.HAND_CONNECTIONS
                )

            # Apply ROI filter if applicable
            if (
                active_effect in ROI_FILTERS
                and left_hand_points
                and right_hand_points
                and len(results_hands.multi_hand_landmarks) == 2
            ):
                rect_points = [
                    left_hand_points[1],
                    right_hand_points[1],
                    right_hand_points[0],
                    left_hand_points[0],
                ]
                pts = np.array(rect_points, np.int32).reshape((-1, 1, 2))
                mask = np.zeros((h, w), dtype=np.uint8)
                cv2.fillPoly(mask, [pts], 255)
                filtered_frame = active_effect(frame)

                inv_mask = cv2.bitwise_not(mask)
                background = cv2.bitwise_and(frame, frame, mask=inv_mask)
                foreground = cv2.bitwise_and(filtered_frame, filtered_frame, mask=mask)
                output_frame = cv2.add(background, foreground)
                cv2.polylines(
                    output_frame, [pts], isClosed=True, color=(0, 255, 0), thickness=2
                )

            # New Bi-directional Filter Switching Logic
            current_time = time.time()
            if current_time - last_pinch_time > PINCH_DEBOUNCE_TIME:
                if right_pinch:
                    current_effect_index = (current_effect_index + 1) % len(all_effects)
                    last_pinch_time = current_time
                elif left_pinch:
                    current_effect_index = (
                        current_effect_index - 1 + len(all_effects)
                    ) % len(all_effects)
                    last_pinch_time = current_time

        cv2.putText(
            output_frame,
            effect_names[current_effect_index],
            (10, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2,
        )
        cv2.imshow(WINDOW_NAME, output_frame)

        # Exit Conditions
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        try:
            if cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
                break
        except cv2.error:
            break

finally:
    print("Cleaning up resources...")
    cap.release()
    cv2.destroyAllWindows()
    hands.close()
    face_mesh.close()
    print("Cleanup complete. Exited gracefully.")
