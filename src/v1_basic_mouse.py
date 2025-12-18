import cv2
import mediapipe as mp
import pyautogui

# -------------------- SETTINGS --------------------
pyautogui.FAILSAFE = False
screen_w, screen_h = pyautogui.size()

SMOOTHING = 5          # Higher = smoother but slower
DEADZONE = 8           # Ignore tiny movements

prev_x, prev_y = screen_w // 2, screen_h // 2

# -------------------- MEDIAPIPE SETUP --------------------
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)
mp_draw = mp.solutions.drawing_utils

# -------------------- CAMERA --------------------
cap = cv2.VideoCapture(0)

print("Gesture Mouse v2 started")

# -------------------- MAIN LOOP --------------------
while True:
    ret, frame = cap.read()
    if not ret:
        print("Camera frame not received")
        break

    frame = cv2.flip(frame, 1)

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    if result.multi_hand_landmarks:
        hand = result.multi_hand_landmarks[0]
        mp_draw.draw_landmarks(frame, hand, mp_hands.HAND_CONNECTIONS)

        # Index finger tip
        index_tip = hand.landmark[8]

        x = int(index_tip.x * screen_w)
        y = int(index_tip.y * screen_h)

        # ---------------- DEAD ZONE ----------------
        if abs(x - prev_x) < DEADZONE and abs(y - prev_y) < DEADZONE:
            x, y = prev_x, prev_y

        # ---------------- SMOOTHING ----------------
        curr_x = prev_x + (x - prev_x) / SMOOTHING
        curr_y = prev_y + (y - prev_y) / SMOOTHING

        pyautogui.moveTo(curr_x, curr_y)

        prev_x, prev_y = curr_x, curr_y

    # ---------------- DISPLAY ----------------
    cv2.imshow("Gesture Mouse v2 - Smooth Cursor (ESC to Exit)", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == 27:   # ESC key
        break

# -------------------- CLEANUP --------------------
cap.release()
cv2.destroyAllWindows()
print("Gesture Mouse v2 ended")
