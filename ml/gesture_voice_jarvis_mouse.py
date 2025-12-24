import cv2
import mediapipe as mp
import pyautogui
import speech_recognition as sr
import threading
import time
import webbrowser
import os
import winsound
from collections import deque

# ================= BASIC SETUP =================
pyautogui.FAILSAFE = False
screen_w, screen_h = pyautogui.size()

# ================= GLOBAL STATES =================
running = True
gesture_enabled = True
wake_word = "alpha"

chat = deque(maxlen=8)

def add_msg(sender, text):
    chat.append(f"{sender}: {text}")

add_msg("SYSTEM", "Say 'alpha' + command")

# ================= VOICE SETUP =================
recognizer = sr.Recognizer()
microphone = sr.Microphone()

def beep():
    winsound.Beep(900, 150)

def voice_listener():
    global running, gesture_enabled

    while running:
        try:
            add_msg("SYSTEM", "Listening...")
            with microphone as source:
                recognizer.adjust_for_ambient_noise(source, duration=0.3)
                audio = recognizer.listen(source, timeout=5)

            beep()
            add_msg("SYSTEM", "Understanding...")
            command = recognizer.recognize_google(audio).lower()
            add_msg("YOU", command)

            # ---------------- WAKE WORD ----------------
            if wake_word not in command:
                add_msg("SYSTEM", "Wake word missing (say 'alpha')")
                continue

            # remove wake word
            command = command.replace(wake_word, "").strip()

            time.sleep(0.4)

            # ---------------- COMMAND EXECUTION ----------------
            if "open" in command and "github" in command:
                add_msg("SYSTEM", "Opening GitHub")
                webbrowser.open("https://github.com")
                add_msg("SYSTEM", "GitHub opened")

            elif "open" in command and "tradingview" in command:
                add_msg("SYSTEM", "Opening TradingView")
                webbrowser.open("https://www.tradingview.com")
                add_msg("SYSTEM", "TradingView opened")

            elif "open" in command and "chrome" in command:
                add_msg("SYSTEM", "Opening Chrome")
                webbrowser.open("https://www.google.com")
                add_msg("SYSTEM", "Chrome opened")

            elif "open" in command and "settings" in command:
                add_msg("SYSTEM", "Opening Settings")
                os.system("start ms-settings:")
                add_msg("SYSTEM", "Settings opened")

            elif ("open" in command and "visual" in command) or ("open" in command and "code" in command):
                add_msg("SYSTEM", "Opening Visual Studio Code")
                os.system("code")
                add_msg("SYSTEM", "VS Code opened")

            elif "search" in command:
                query = command.replace("search", "").strip()
                if query:
                    add_msg("SYSTEM", f"Searching {query}")
                    webbrowser.open(f"https://www.google.com/search?q={query}")
                    add_msg("SYSTEM", "Search completed")
                else:
                    add_msg("SYSTEM", "Search keyword missing")

            elif "start" in command and "control" in command:
                gesture_enabled = True
                add_msg("SYSTEM", "Gesture control enabled")

            elif "stop" in command and "control" in command:
                gesture_enabled = False
                add_msg("SYSTEM", "Gesture control disabled")

            elif "exit" in command:
                add_msg("SYSTEM", "Exiting system")
                running = False

            else:
                add_msg("SYSTEM", "Command not recognized")

            time.sleep(0.6)

        except sr.WaitTimeoutError:
            add_msg("SYSTEM", "No speech detected")

        except sr.UnknownValueError:
            add_msg("SYSTEM", "Could not understand")

        except Exception as e:
            add_msg("SYSTEM", "Mic error")

# ================= START VOICE THREAD =================
threading.Thread(target=voice_listener, daemon=True).start()

# ================= MEDIAPIPE =================
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1)
mp_draw = mp.solutions.drawing_utils
cap = cv2.VideoCapture(0)

# ================= CURSOR CONTROL =================
cursor_x, cursor_y = screen_w // 2, screen_h // 2
MOVE_ALPHA = 0.05
DEADZONE = 40
dragging = False

def finger_open(lm, tip, pip):
    return lm[tip].y < lm[pip].y

# ================= MAIN LOOP =================
while running:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    if gesture_enabled and result.multi_hand_landmarks:
        lm = result.multi_hand_landmarks[0].landmark
        mp_draw.draw_landmarks(frame, result.multi_hand_landmarks[0], mp_hands.HAND_CONNECTIONS)

        index_open = finger_open(lm, 8, 6)
        middle_open = finger_open(lm, 12, 10)
        ring_open = finger_open(lm, 16, 14)
        pinky_open = finger_open(lm, 20, 18)

        # Cursor move
        if index_open and middle_open:
            ix = lm[8].x * screen_w
            iy = lm[8].y * screen_h
            dx, dy = ix - cursor_x, iy - cursor_y
            if abs(dx) > DEADZONE or abs(dy) > DEADZONE:
                cursor_x += dx * MOVE_ALPHA
                cursor_y += dy * MOVE_ALPHA
                pyautogui.moveTo(cursor_x, cursor_y)

        # Drag & drop
        if not index_open and not middle_open and not ring_open and not pinky_open:
            if not dragging:
                pyautogui.mouseDown()
                dragging = True
        else:
            if dragging:
                pyautogui.mouseUp()
                dragging = False

    # ================= CHAT BOX =================
    cv2.rectangle(frame, (0, 0), (760, 240), (0, 0, 0), -1)
    y = 25
    for msg in chat:
        color = (0, 255, 0) if msg.startswith("SYSTEM") else (255, 255, 255)
        cv2.putText(frame, msg, (10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        y += 28

    cv2.imshow("Gesture + Voice Assistant (Alpha)", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

running = False
cap.release()
cv2.destroyAllWindows()
