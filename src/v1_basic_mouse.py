import cv2
import mediapipe as mp
import pyautogui
import time

pyautogui.FAILSAFE = False
screen_w, screen_h = pyautogui.size()

# ================= PARAMETERS =================
SMOOTH_ALPHA = 0.2
DEADZONE = 12
CLICK_COOLDOWN = 0.6

# ================= STATES =================
MOVE = "MOVE"
LEFT_LOCK = "LEFT_LOCK"
RIGHT_LOCK = "RIGHT_LOCK"
DRAG_LOCK = "DRAG_LOCK"

state = MOVE
last_action_time = 0

# HARD cursor anchor (absolute lock)
cursor_x = screen_w // 2
cursor_y = screen_h // 2

# ================= MEDIAPIPE =================
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

# ================= HELPERS =================
def finger_open(lm, tip, pip):
    return lm[tip].y < lm[pip].y

def all_closed(lm):
    return all(lm[t].y > lm[p].y for t,p in [(8,6),(12,10),(16,14),(20,18)])

def all_open(lm):
    return all(lm[t].y < lm[p].y for t,p in [(8,6),(12,10),(16,14),(20,18)])

# ================= MAIN LOOP =================
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    if result.multi_hand_landmarks:
        hand = result.multi_hand_landmarks[0]
        mp_draw.draw_landmarks(frame, hand, mp_hands.HAND_CONNECTIONS)
        lm = hand.landmark

        index_open  = finger_open(lm, 8, 6)
        middle_open = finger_open(lm,12,10)

        now = time.time()

        # ================= LEFT CLICK (INDEX FOLDED) =================
        if (
            state == MOVE and
            not index_open and middle_open and
            now - last_action_time > CLICK_COOLDOWN
        ):
            state = LEFT_LOCK
            pyautogui.moveTo(cursor_x, cursor_y)
            pyautogui.click()
            last_action_time = now

        # ================= RIGHT CLICK (MIDDLE FOLDED) =================
        elif (
            state == MOVE and
            index_open and not middle_open and
            now - last_action_time > CLICK_COOLDOWN
        ):
            state = RIGHT_LOCK
            pyautogui.moveTo(cursor_x, cursor_y)
            pyautogui.rightClick()
            last_action_time = now

        # ================= EXIT CLICK LOCK =================
        elif state in [LEFT_LOCK, RIGHT_LOCK] and index_open and middle_open:
            state = MOVE

        # ================= DRAG =================
        elif state == MOVE and all_closed(lm):
            state = DRAG_LOCK
            pyautogui.moveTo(cursor_x, cursor_y)
            pyautogui.mouseDown()

        elif state == DRAG_LOCK and all_open(lm):
            pyautogui.mouseUp()
            state = MOVE

        # ================= MOVE (ONLY HERE) =================
        if state == MOVE and index_open and middle_open:
            ix = lm[8].x * screen_w
            iy = lm[8].y * screen_h

            dx = ix - cursor_x
            dy = iy - cursor_y

            if abs(dx) > DEADZONE or abs(dy) > DEADZONE:
                cursor_x += dx * SMOOTH_ALPHA
                cursor_y += dy * SMOOTH_ALPHA
                pyautogui.moveTo(cursor_x, cursor_y)

        # ================= HARD LOCK ENFORCEMENT =================
        if state in [LEFT_LOCK, RIGHT_LOCK, DRAG_LOCK]:
            pyautogui.moveTo(cursor_x, cursor_y)

    cv2.imshow("Hand Mouse – HARD FREEZE MODE (ESC)", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
