# hand_mouse_final_calibrated.py
# Final (calibrated from user's close-photo)
# Cursor ON only when index+middle are very close (A: strict)
# Requirements: Python 3.10, mediapipe, opencv-python, pyautogui, numpy

import cv2
import mediapipe as mp
import pyautogui
import math, time
import numpy as np
from collections import deque

pyautogui.FAILSAFE = False

# ----------------- CALIBRATED TUNABLES -----------------
# VERY CLOSE threshold (calibrated from user's photo at /mnt/data/WIN_20251119_15_19_24_Pro.jpg)
CURSOR_GAP_VERY_CLOSE = 0.035
 # normalized (0..1). Lower = stricter; increase if cursor never activates.

# Cursor movement
SMOOTH_ALPHA = 0.12
DEADZONE_PIX = 6

# Adaptive mapping
HIST_LEN = 450
MIN_CALIB = 16
LOW_PER = 4
HIGH_PER = 96
EXPAND_BOX = 0.90
EDGE_POWER = 1.03

# Click/drag/scroll tuning
INDEX_FOLD_FRAMES = 3
INDEX_FOLD_DEBOUNCE = 0.20
RIGHT_FOLD_FRAMES = 3
RIGHT_CLICK_DEBOUNCE = 0.45

SCROLL_MIN_DELTA = 0.004
SCROLL_SENS = 300

DRAG_FRAMES_REQ = 6
DRAG_SMOOTH = 0.03     # very slow anchor update for pixel-perfect drag
DRAG_MAX_STEP = 2.0

ANCHOR_HIST_LEN = 12
ANCHOR_JUMP_THRESH = 0.06
ANCHOR_STABLE_REQ = 2

CAM_W, CAM_H = 640, 480
# ------------------------------------------------------

screen_w, screen_h = pyautogui.size()
prev_x, prev_y = screen_w/2.0, screen_h/2.0

hist_x = deque(maxlen=HIST_LEN)
hist_y = deque(maxlen=HIST_LEN)
anchor_hist = deque(maxlen=ANCHOR_HIST_LEN)

anchor_stable_frames = 0
drag_locked_pos = None
is_dragging = False
drag_frame_count = 0
drag_just_started = False

index_fold_count = 0
right_fold_count = 0
last_left_time = 0.0
last_right_time = 0.0
scroll_anchor_y = None
show_skeleton = False

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

def ndist(a,b): return math.hypot(a.x - b.x, a.y - b.y)
def finger_up(lm, tip, pip): return lm[tip].y < lm[pip].y
def finger_fold(lm, tip, pip): return lm[tip].y > lm[pip].y

def remap_edge(v, lo, hi, p):
    if hi - lo == 0:
        t = 0.5
    else:
        t = (v - lo) / (hi - lo)
    t = max(0.0, min(1.0, t))
    t0 = 2 * (t - 0.5)
    t1 = np.sign(t0) * (abs(t0) ** p)
    return float((t1 + 1) / 2)

# Camera init
cap = cv2.VideoCapture(0)
cap.set(3, CAM_W)
cap.set(4, CAM_H)
time.sleep(0.2)

with mp_hands.Hands(max_num_hands=1, model_complexity=1,
                    min_detection_confidence=0.72, min_tracking_confidence=0.72) as hands:

    print("Hand Mouse - Final calibrated (A strict).")
    print("Reference image used:", "/mnt/data/WIN_20251119_15_19_24_Pro.jpg")
    print("ESC to quit | V to toggle skeleton overlay")

    while True:
        ok, frame = cap.read()
        if not ok:
            print("Camera frame not received.")
            break

        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = hands.process(img)

        key = cv2.waitKey(1) & 0xFF
        if key == 27: break
        if key == ord('v'): show_skeleton = not show_skeleton

        gesture_text = ""
        cursor_active = False

        if res.multi_hand_landmarks:
            hand = res.multi_hand_landmarks[0]
            lm = hand.landmark

            if show_skeleton:
                mp_draw.draw_landmarks(frame, hand, mp_hands.HAND_CONNECTIONS)

            # landmarks shortcuts
            it = lm[8]; mt = lm[12]; rt = lm[16]; pt = lm[20]
            thumb = lm[4]; palm = lm[9]

            # finger states
            index_up = finger_up(lm,8,6)
            middle_up = finger_up(lm,12,10)
            ring_down = finger_fold(lm,16,14)
            pinky_down = finger_fold(lm,20,18)

            # normalized distance between index and middle
            im_dist = ndist(it, mt)

            # A strict rule: very close required
            fingers_close = im_dist < CURSOR_GAP_VERY_CLOSE

            # cursor allowed posture
            cursor_allowed = index_up and middle_up and ring_down and pinky_down and fingers_close

            # midpoint anchor
            ax = (it.x + mt.x) / 2.0
            ay = (it.y + mt.y) / 2.0

            # build history (only when both extended)
            if index_up and middle_up:
                hist_x.append(ax); hist_y.append(ay)

            # adaptive mapping
            if len(hist_x) >= MIN_CALIB:
                xs = np.array(hist_x); ys = np.array(hist_y)
                lo_x = float(np.percentile(xs, LOW_PER))
                hi_x = float(np.percentile(xs, HIGH_PER))
                lo_y = float(np.percentile(ys, LOW_PER))
                hi_y = float(np.percentile(ys, HIGH_PER))

                lo_x = max(0.0, lo_x - EXPAND_BOX)
                hi_x = min(1.0, hi_x + EXPAND_BOX)
                lo_y = max(0.0, lo_y - EXPAND_BOX)
                hi_y = min(1.0, hi_y + EXPAND_BOX)

                # safety widen if too narrow
                if hi_x - lo_x < 0.02:
                    lo_x = max(0, lo_x - 0.05); hi_x = min(1, hi_x + 0.05)
                if hi_y - lo_y < 0.02:
                    lo_y = max(0, lo_y - 0.05); hi_y = min(1, hi_y + 0.05)

                sx = remap_edge(ax, lo_x, hi_x, EDGE_POWER)
                sy = remap_edge(ay, lo_y, hi_y, EDGE_POWER)
            else:
                sx = remap_edge(ax, 0.15, 0.85, EDGE_POWER)
                sy = remap_edge(ay, 0.15, 0.85, EDGE_POWER)

            target_x = sx * screen_w
            target_y = sy * screen_h

            # anchor median + jump filter (for drag)
            anchor_hist.append((ax, ay))
            arr = np.array(anchor_hist)
            med_ax = float(np.median(arr[:,0])); med_ay = float(np.median(arr[:,1]))
            jump = math.hypot(ax - med_ax, ay - med_ay)
            if jump > ANCHOR_JUMP_THRESH:
                anchor_stable_frames = 0
                eff_ax, eff_ay = med_ax, med_ay
            else:
                anchor_stable_frames += 1
                eff_ax, eff_ay = ax, ay
            if anchor_stable_frames >= ANCHOR_STABLE_REQ:
                eff_ax, eff_ay = med_ax, med_ay

            # fist detection
            avg_tip_to_palm = (ndist(it,palm) + ndist(mt,palm) + ndist(rt,palm) + ndist(pt,palm)) / 4.0
            fingers_folded = (not index_up) and (not middle_up) and finger_fold(lm,16,14) and finger_fold(lm,20,18)
            is_fist = fingers_folded and (avg_tip_to_palm < 0.11)

            open_hand = finger_up(lm,8,6) and finger_up(lm,12,10) and finger_up(lm,16,14) and finger_up(lm,20,18)

            # ---------- Cursor movement ----------
            if not is_dragging:
                if cursor_allowed:
                    cursor_active = True
                    smooth_x = prev_x + (target_x - prev_x) * SMOOTH_ALPHA
                    smooth_y = prev_y + (target_y - prev_y) * SMOOTH_ALPHA
                    if abs(smooth_x - prev_x) > DEADZONE_PIX or abs(smooth_y - prev_y) > DEADZONE_PIX:
                        pyautogui.moveTo(int(smooth_x), int(smooth_y))
                        prev_x, prev_y = smooth_x, smooth_y
                else:
                    cursor_active = False
            else:
                # DRAG MODE — very slow precise follow of anchor (eff_ax/eff_ay)
                raw_x = eff_ax * screen_w
                raw_y = eff_ay * screen_h

                if drag_just_started and drag_locked_pos is None:
                    drag_locked_pos = [raw_x, raw_y]
                    drag_just_started = False

                lx, ly = drag_locked_pos
                dx = raw_x - lx; dy = raw_y - ly

                if math.hypot(dx/screen_w, dy/screen_h) > ANCHOR_JUMP_THRESH:
                    new_x, new_y = lx, ly
                else:
                    new_x = lx + dx * DRAG_SMOOTH
                    new_y = ly + dy * DRAG_SMOOTH

                if abs(new_x - lx) > DRAG_MAX_STEP:
                    new_x = lx + math.copysign(DRAG_MAX_STEP, new_x - lx)
                if abs(new_y - ly) > DRAG_MAX_STEP:
                    new_y = ly + math.copysign(DRAG_MAX_STEP, new_y - ly)

                pyautogui.moveTo(int(new_x), int(new_y))
                drag_locked_pos = [new_x, new_y]
                prev_x, prev_y = new_x, new_y

            # visual indicator
            color = (0,200,0) if cursor_active else (0,60,200)
            cv2.circle(frame, (int(ax*w), int(ay*h)), 7, color, -1)
            cv2.putText(frame, "ACTIVE" if cursor_active else "INACTIVE",
                        (int(ax*w)+10, int(ay*h)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

            # ---------- LEFT CLICK (index fold) ----------
            if (not index_up) and middle_up and not is_dragging:
                index_fold_count += 1
            else:
                index_fold_count = 0
            if index_fold_count >= INDEX_FOLD_FRAMES and (time.time() - last_left_time > INDEX_FOLD_DEBOUNCE):
                pyautogui.click()
                last_left_time = time.time()
                index_fold_count = 0
                gesture_text = "LEFT CLICK"

            # ---------- RIGHT CLICK (middle fold) ----------
            if (not middle_up) and index_up and not is_dragging:
                right_fold_count += 1
            else:
                right_fold_count = 0
            if right_fold_count >= RIGHT_FOLD_FRAMES and (time.time() - last_right_time > RIGHT_CLICK_DEBOUNCE):
                pyautogui.rightClick()
                last_right_time = time.time()
                right_fold_count = 0
                gesture_text = "RIGHT CLICK"

            # ---------- SCROLL (four fingers up) ----------
            all_up = finger_up(lm,8,6) and finger_up(lm,12,10) and finger_up(lm,16,14) and finger_up(lm,20,18)
            if all_up:
                cur_avg_y = (it.y + mt.y + rt.y + pt.y) / 4.0
                if scroll_anchor_y is None:
                    scroll_anchor_y = cur_avg_y
                else:
                    dy_norm = scroll_anchor_y - cur_avg_y
                    if abs(dy_norm) > SCROLL_MIN_DELTA:
                        pyautogui.scroll(int(dy_norm * SCROLL_SENS))
                        gesture_text = "SCROLL"
                    scroll_anchor_y = scroll_anchor_y * 0.85 + cur_avg_y * 0.15
            else:
                scroll_anchor_y = None

            # ---------- DRAG START ----------
            if is_fist and not is_dragging:
                drag_frame_count += 1
                if drag_frame_count >= DRAG_FRAMES_REQ:
                    snap_x = int(eff_ax * screen_w)
                    snap_y = int(eff_ay * screen_h)
                    pyautogui.moveTo(snap_x, snap_y)
                    time.sleep(0.03)
                    pyautogui.mouseDown()
                    is_dragging = True
                    drag_just_started = True
                    drag_locked_pos = [snap_x, snap_y]
                    drag_frame_count = 0
                    gesture_text = "DRAG START"
            else:
                if not is_fist:
                    drag_frame_count = 0

            # ---------- DROP ----------
            if is_dragging and open_hand:
                pyautogui.mouseUp()
                is_dragging = False
                drag_locked_pos = None
                gesture_text = "DROP"

        else:
            # no hand -> safe cleanup / release drag
            scroll_anchor_y = None
            drag_frame_count = 0
            drag_just_started = False
            anchor_hist.clear()
            if is_dragging:
                pyautogui.mouseUp()
                is_dragging = False
                drag_locked_pos = None

        # Draw UI
        if gesture_text:
            cv2.putText(frame, gesture_text, (8,44), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
        cv2.putText(frame, "ESC quit | V toggle skeleton", (8,18), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (220,220,220), 2)
        cv2.imshow("Hand Mouse Final - Calibrated A strict", frame)

    # cleanup
cap.release()
cv2.destroyAllWindows()
