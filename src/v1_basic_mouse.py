import cv2
import mediapipe as mp
import pyautogui

print("Gesture Mouse v1 started")

pyautogui.FAILSAFE = False
screen_w, screen_h = pyautogui.size()

# MediaPipe setup
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    if result.multi_hand_landmarks:
        hand = result.multi_hand_landmarks[0]
        mp_draw.draw_landmarks(frame, hand, mp_hands.HAND_CONNECTIONS)

        index_tip = hand.landmark[8]
        x = int(index_tip.x * screen_w)
        y = int(index_tip.y * screen_h)

        pyautogui.moveTo(x, y)

    cv2.imshow("Gesture Mouse v1 - Cursor Only (ESC to exit)", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
print("Gesture Mouse v1 ended")
