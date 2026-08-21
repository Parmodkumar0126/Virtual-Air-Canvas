import os
import time
import datetime
import urllib.request
import random
import math
from collections import deque

import cv2
import mediapipe as mp
import numpy as np

from mediapipe.tasks import python
from mediapipe.tasks.python import vision


# =========================================================
# SETTINGS
# =========================================================

MODEL_PATH = "hand_landmarker.task"

MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/1/hand_landmarker.task"
)

FRAME_WIDTH = 1280
FRAME_HEIGHT = 720

PARTICLE_COUNT = 220

WINDOW_NAME = "Virtual Air Canvas - AI Vision"

# Tracking stability
TRACK_GRACE_FRAMES = 10
TRACK_MATCH_DISTANCE = 180

# Screenshot
SCREENSHOT_HOLD_SECONDS = 1.0
SCREENSHOT_PALM_DISTANCE = 220
SCREENSHOT_COOLDOWN_SECONDS = 2.0
SCREENSHOT_DIR = "screenshots"

# FPS smoothing
FPS_WINDOW = 15

# Blur performance: how much to shrink a layer before blurring,
# then scale the blur back up. 0.5 = half resolution blur pass.
BLUR_DOWNSCALE = 0.5


# =========================================================
# COLORS - BGR
# =========================================================

CYAN = (255, 255, 0)
BLUE = (255, 80, 0)
PURPLE = (255, 0, 180)
RED = (0, 0, 255)
YELLOW = (0, 255, 255)
WHITE = (255, 255, 255)
GREEN = (0, 255, 0)


# =========================================================
# HAND CONNECTIONS
# =========================================================

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17)
]

FINGERTIP_IDS = [4, 8, 12, 16, 20]
INDEX_TIP = 8


# =========================================================
# MODEL
# =========================================================

def ensure_model():

    if os.path.exists(MODEL_PATH):
        return

    print("Downloading hand_landmarker.task...")

    try:
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        print("Model downloaded successfully.")

    except Exception as e:
        print("Model download failed:")
        print(e)
        raise SystemExit(1)


def get_detector():

    base_options = python.BaseOptions(model_asset_path=MODEL_PATH)

    options = vision.HandLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.VIDEO,
        num_hands=2,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5
    )

    return vision.HandLandmarker.create_from_options(options)


# =========================================================
# LANDMARKS
# =========================================================

def landmarks_to_points(hand, width, height):

    points = []

    for lm in hand:
        x = int(lm.x * width)
        y = int(lm.y * height)

        x = max(0, min(width - 1, x))
        y = max(0, min(height - 1, y))

        points.append((x, y))

    return points


# =========================================================
# FAST GLOW BLUR (downscale/blur/upscale for speed)
# =========================================================

def fast_glow(layer, sigma):

    h, w = layer.shape[:2]

    small_w = max(1, int(w * BLUR_DOWNSCALE))
    small_h = max(1, int(h * BLUR_DOWNSCALE))

    small = cv2.resize(layer, (small_w, small_h), interpolation=cv2.INTER_LINEAR)
    small_blur = cv2.GaussianBlur(small, (0, 0), sigma * BLUR_DOWNSCALE)

    return cv2.resize(small_blur, (w, h), interpolation=cv2.INTER_LINEAR)


# =========================================================
# HAND TRACKER
# =========================================================

class HandTracker:

    def __init__(self):
        self.tracks = []

    def update(self, current_hands):

        unmatched = list(current_hands)

        for track in self.tracks:

            wrist = track["points"][0]

            best_match = None
            best_dist = TRACK_MATCH_DISTANCE

            for hand in unmatched:
                dist = math.hypot(hand[0][0] - wrist[0], hand[0][1] - wrist[1])

                if dist < best_dist:
                    best_dist = dist
                    best_match = hand

            if best_match is not None:
                track["points"] = best_match
                track["lost"] = 0
                unmatched.remove(best_match)
            else:
                track["lost"] += 1

        self.tracks = [t for t in self.tracks if t["lost"] <= TRACK_GRACE_FRAMES]

        for hand in unmatched:
            if len(self.tracks) < 2:
                self.tracks.append({"points": hand, "lost": 0})

        return [t["points"] for t in self.tracks]

    def fresh_hands(self, current_hands, tracked_hands):
        fresh = []
        for hand in tracked_hands:
            if hand in current_hands:
                fresh.append(hand)
        return fresh


# =========================================================
# HAND SKELETON
# =========================================================

def draw_hand_skeleton(layer, points):

    for start, end in HAND_CONNECTIONS:
        cv2.line(layer, points[start], points[end], WHITE, 2, cv2.LINE_AA)

    for i, point in enumerate(points):
        if i in FINGERTIP_IDS:
            cv2.circle(layer, point, 6, YELLOW, -1, cv2.LINE_AA)
            cv2.circle(layer, point, 10, WHITE, 1, cv2.LINE_AA)
        else:
            cv2.circle(layer, point, 4, RED, -1, cv2.LINE_AA)


# =========================================================
# TWO HAND WEB
# =========================================================

def draw_dual_hand_web(glow_layer, line_layer, hand1, hand2):

    for i in range(21):
        p1 = hand1[i]
        p2 = hand2[i]

        cv2.line(glow_layer, p1, p2, CYAN, 10, cv2.LINE_AA)
        cv2.line(line_layer, p1, p2, CYAN, 2, cv2.LINE_AA)

    for index, lid in enumerate(FINGERTIP_IDS):
        p1 = hand1[lid]
        p2 = hand2[lid]

        if index % 3 == 0:
            color = CYAN
        elif index % 3 == 1:
            color = BLUE
        else:
            color = PURPLE

        cv2.line(glow_layer, p1, p2, color, 18, cv2.LINE_AA)
        cv2.line(line_layer, p1, p2, color, 3, cv2.LINE_AA)


# =========================================================
# FINGER / GESTURE DETECTION
# =========================================================

def finger_is_up(points, tip_id, pip_id, mcp_id):

    wrist = np.array(points[0])
    tip = np.array(points[tip_id])
    pip = np.array(points[pip_id])

    tip_distance = np.linalg.norm(tip - wrist)
    pip_distance = np.linalg.norm(pip - wrist)

    return tip_distance > pip_distance * 1.15


def get_gesture(points):

    index_up = finger_is_up(points, 8, 6, 5)
    middle_up = finger_is_up(points, 12, 10, 9)
    ring_up = finger_is_up(points, 16, 14, 13)
    pinky_up = finger_is_up(points, 20, 18, 17)

    if index_up and not middle_up and not ring_up and not pinky_up:
        return 1
    if index_up and middle_up and not ring_up and not pinky_up:
        return 2
    if index_up and middle_up and ring_up and not pinky_up:
        return 3
    if index_up and middle_up and ring_up and pinky_up:
        return 4

    return 0


# =========================================================
# PARTICLE
# =========================================================

class Particle:

    def __init__(self, width, height):
        self.x = random.uniform(0, width)
        self.y = random.uniform(0, height)
        self.vx = random.uniform(-1, 1)
        self.vy = random.uniform(-1, 1)
        self.size = random.randint(1, 3)

    def update(self, width, height, hands, gesture):

        self.x += self.vx
        self.y += self.vy

        if gesture == 0:
            self.vx *= 0.995
            self.vy *= 0.995

        elif gesture == 1 and hands:
            tx, ty = hands[0][INDEX_TIP]
            dx = tx - self.x
            dy = ty - self.y
            distance = math.hypot(dx, dy)

            if distance > 5:
                force = min(0.08, 150 / (distance + 1))
                self.vx += dx / distance * force
                self.vy += dy / distance * force

        elif gesture == 2 and hands:
            for hand in hands:
                tx, ty = hand[INDEX_TIP]
                dx = tx - self.x
                dy = ty - self.y
                distance = math.hypot(dx, dy)

                if 10 < distance < 300:
                    force = 0.04
                    self.vx += dx / distance * force
                    self.vy += dy / distance * force

        elif gesture == 3 and hands:
            tx, ty = hands[0][INDEX_TIP]
            dx = self.x - tx
            dy = self.y - ty
            distance = math.hypot(dx, dy)

            if 1 < distance < 300:
                force = 1.8 / (distance / 30 + 1)
                self.vx += dx / distance * force
                self.vy += dy / distance * force

        elif gesture == 4 and hands:
            tx, ty = hands[0][0]
            dx = tx - self.x
            dy = ty - self.y
            distance = math.hypot(dx, dy)

            if distance > 5:
                force = 0.05
                self.vx += dx / distance * force
                self.vy += dy / distance * force

        speed = math.hypot(self.vx, self.vy)

        if speed > 5:
            self.vx = self.vx / speed * 5
            self.vy = self.vy / speed * 5

        if self.x < 0:
            self.x = width
        if self.x > width:
            self.x = 0
        if self.y < 0:
            self.y = height
        if self.y > height:
            self.y = 0


def create_particles(count, width, height):
    return [Particle(width, height) for _ in range(count)]


def draw_particles(layer, particles, color):
    for particle in particles:
        cv2.circle(layer, (int(particle.x), int(particle.y)), particle.size, color, -1, cv2.LINE_AA)


def connect_particles(layer, particles, color):
    for i in range(0, len(particles), 3):
        p1 = particles[i]

        for j in range(i + 3, min(i + 25, len(particles)), 3):
            p2 = particles[j]

            distance = math.hypot(p1.x - p2.x, p1.y - p2.y)

            if distance < 100:
                cv2.line(layer, (int(p1.x), int(p1.y)), (int(p2.x), int(p2.y)), color, 1, cv2.LINE_AA)


# =========================================================
# HOLOGRAPHIC BUTTON / PANEL
# =========================================================

class HoloButton:

    def __init__(self, x, y, width, height, text, color):
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.text = text
        self.color = color

    def contains(self, px, py):
        return self.x <= px <= self.x + self.width and self.y <= py <= self.y + self.height


# Buttons are static (same position every frame) - build once, reuse.
_PANEL_X = 30
_PANEL_Y = 105
_BUTTON_W = 145
_BUTTON_H = 48
_BUTTON_GAP = 10

_PANEL_BUTTONS = [
    HoloButton(_PANEL_X, _PANEL_Y, _BUTTON_W, _BUTTON_H, "PARTICLES", CYAN),
    HoloButton(_PANEL_X + (_BUTTON_W + _BUTTON_GAP), _PANEL_Y, _BUTTON_W, _BUTTON_H, "WEB", PURPLE),
    HoloButton(_PANEL_X + 2 * (_BUTTON_W + _BUTTON_GAP), _PANEL_Y, _BUTTON_W, _BUTTON_H, "AURA", GREEN),
    HoloButton(_PANEL_X + 3 * (_BUTTON_W + _BUTTON_GAP), _PANEL_Y, _BUTTON_W, _BUTTON_H, "CLOCK", YELLOW),
    HoloButton(_PANEL_X + 4 * (_BUTTON_W + _BUTTON_GAP), _PANEL_Y, _BUTTON_W, _BUTTON_H, "RESET", RED)
]

# Bounding box that covers the whole panel (used for a small ROI copy
# instead of copying the entire frame every frame).
_PANEL_ROI_X0 = _PANEL_X - 15
_PANEL_ROI_Y0 = _PANEL_Y - 40
_PANEL_ROI_X1 = _PANEL_X + 5 * (_BUTTON_W + _BUTTON_GAP)
_PANEL_ROI_Y1 = _PANEL_Y + _BUTTON_H + 10


def draw_holo_panel(frame, finger, selected_mode, glow_scratch):

    buttons = _PANEL_BUTTONS

    h, w = frame.shape[:2]
    x0 = max(0, _PANEL_ROI_X0)
    y0 = max(0, _PANEL_ROI_Y0)
    x1 = min(w, _PANEL_ROI_X1)
    y1 = min(h, _PANEL_ROI_Y1)

    # Only copy the small panel region, not the full frame.
    roi = frame[y0:y1, x0:x1]
    overlay = roi.copy()

    def shift(px, py):
        return (px - x0, py - y0)

    panel_x, panel_y = shift(_PANEL_X, _PANEL_Y)

    cv2.putText(overlay, "HOLOGRAPHIC CONTROL", (panel_x, panel_y - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.48, CYAN, 1, cv2.LINE_AA)

    hovered = None

    if finger is not None:
        fx, fy = finger
        for button in buttons:
            if button.contains(fx, fy):
                hovered = button.text

    for button in buttons:

        bx, by = shift(button.x, button.y)

        is_hovered = (hovered == button.text)
        is_selected = (selected_mode == button.text)

        if is_hovered or is_selected:

            cv2.rectangle(
                overlay,
                (bx - 3, by - 3),
                (bx + button.width + 3, by + button.height + 3),
                button.color, 3, cv2.LINE_AA
            )

            # Reuse a full-size scratch buffer for the glow instead of
            # allocating a new np.zeros_like(frame) every frame.
            glow_scratch.fill(0)

            cv2.rectangle(
                glow_scratch,
                (button.x, button.y),
                (button.x + button.width, button.y + button.height),
                button.color, -1
            )

            glow = fast_glow(glow_scratch, 12)

            frame[:] = cv2.addWeighted(frame, 1.0, glow, 0.18, 0)

        else:
            cv2.rectangle(
                overlay,
                (bx, by),
                (bx + button.width, by + button.height),
                (80, 80, 80), 1, cv2.LINE_AA
            )

        text_size = cv2.getTextSize(button.text, cv2.FONT_HERSHEY_SIMPLEX, 0.43, 1)[0]

        text_x = bx + (button.width - text_size[0]) // 2
        text_y = by + (button.height + text_size[1]) // 2

        cv2.putText(
            overlay, button.text, (text_x, text_y),
            cv2.FONT_HERSHEY_SIMPLEX, 0.43,
            button.color if (is_hovered or is_selected) else WHITE,
            1, cv2.LINE_AA
        )

    roi[:] = cv2.addWeighted(roi, 0.72, overlay, 0.28, 0)

    if finger is not None:
        fx, fy = finger
        cv2.circle(frame, (fx, fy), 14, CYAN, 2, cv2.LINE_AA)
        cv2.circle(frame, (fx, fy), 4, WHITE, -1, cv2.LINE_AA)

    return hovered


# =========================================================
# HAND AURA
# =========================================================

def draw_hand_aura(glow_layer, hand_points, color, animation_time):

    palm = hand_points[0]
    x, y = palm

    for i in range(5):
        radius = 40 + i * 13 + int(5 * math.sin(animation_time * 3 + i))
        cv2.circle(glow_layer, (x, y), radius, color, 2, cv2.LINE_AA)


# =========================================================
# HOLOGRAPHIC SPHERE
# =========================================================

def draw_holographic_sphere(frame, glow_layer, hand1, hand2, animation_time):

    p1 = np.array(hand1[0], dtype=np.float32)
    p2 = np.array(hand2[0], dtype=np.float32)

    distance = float(np.linalg.norm(p1 - p2))

    center_x = int((p1[0] + p2[0]) / 2)
    center_y = int((p1[1] + p2[1]) / 2)

    radius = int(np.clip(distance * 0.30, 35, 145))

    cv2.circle(glow_layer, (center_x, center_y), radius + 18, CYAN, 8, cv2.LINE_AA)
    cv2.circle(glow_layer, (center_x, center_y), radius, PURPLE, 5, cv2.LINE_AA)

    for i in range(4):
        extra = int(8 * math.sin(animation_time * 2 + i))
        current_radius = radius - i * 7 + extra

        if current_radius > 5:
            cv2.circle(
                frame, (center_x, center_y), current_radius,
                CYAN if i % 2 == 0 else PURPLE, 1, cv2.LINE_AA
            )

    angle = animation_time * 2

    for i in range(6):
        current_angle = angle + i * math.pi / 3

        x1 = int(center_x + radius * math.cos(current_angle))
        y1 = int(center_y + radius * math.sin(current_angle))
        x2 = int(center_x - radius * math.cos(current_angle))
        y2 = int(center_y - radius * math.sin(current_angle))

        cv2.line(frame, (x1, y1), (x2, y2), CYAN, 1, cv2.LINE_AA)

    orbit_radius = radius + 20
    orbit_angle = animation_time * 2.5

    orbit_x = int(center_x + orbit_radius * math.cos(orbit_angle))
    orbit_y = int(center_y + orbit_radius * math.sin(orbit_angle))

    cv2.circle(frame, (orbit_x, orbit_y), 6, YELLOW, -1, cv2.LINE_AA)

    cv2.line(glow_layer, (int(p1[0]), int(p1[1])), (center_x, center_y), CYAN, 5, cv2.LINE_AA)
    cv2.line(glow_layer, (int(p2[0]), int(p2[1])), (center_x, center_y), PURPLE, 5, cv2.LINE_AA)

    return distance, radius


def draw_sphere_particles(layer, center, radius, animation_time):

    cx, cy = center
    particle_count = 28

    for i in range(particle_count):
        angle = animation_time * 1.5 + i * (2 * math.pi / particle_count)
        orbit = radius + 18 + 12 * math.sin(animation_time + i)

        x = int(cx + orbit * math.cos(angle))
        y = int(cy + orbit * math.sin(angle) * 0.45)

        cv2.circle(layer, (x, y), 2, CYAN if i % 2 == 0 else PURPLE, -1, cv2.LINE_AA)


# =========================================================
# REAL-TIME HOLOGRAPHIC CLOCK
# =========================================================

def draw_holographic_clock(frame, center, sphere_radius):

    now = datetime.datetime.now()
    cx, cy = center

    clock_radius = int(np.clip(sphere_radius - 12, 28, 78))

    h, w = frame.shape[:2]
    x0 = max(0, cx - clock_radius - 25)
    y0 = max(0, cy - clock_radius - 25)
    x1 = min(w, cx + clock_radius + 25)
    y1 = min(h, cy + clock_radius + 45)

    if x1 <= x0 or y1 <= y0:
        return

    roi = frame[y0:y1, x0:x1]
    overlay = roi.copy()

    lcx, lcy = cx - x0, cy - y0

    cv2.circle(overlay, (lcx, lcy), clock_radius, (8, 12, 20), -1, cv2.LINE_AA)

    roi[:] = cv2.addWeighted(roi, 0.35, overlay, 0.65, 0)

    cv2.circle(frame, (cx, cy), clock_radius, CYAN, 2, cv2.LINE_AA)

    if clock_radius > 35:
        cv2.circle(frame, (cx, cy), clock_radius - 5, PURPLE, 1, cv2.LINE_AA)

    numbers = [
        ("12", 0), ("1", 30), ("2", 60), ("3", 90), ("4", 120), ("5", 150),
        ("6", 180), ("7", 210), ("8", 240), ("9", 270), ("10", 300), ("11", 330)
    ]

    number_radius = clock_radius - 14
    font_scale = 0.32 if clock_radius < 45 else 0.38

    for text, degree in numbers:
        angle = math.radians(degree - 90)

        x = int(cx + number_radius * math.cos(angle))
        y = int(cy + number_radius * math.sin(angle))

        text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)[0]

        x -= text_size[0] // 2
        y += text_size[1] // 2

        cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, WHITE, 1, cv2.LINE_AA)

    hour = now.hour % 12
    minute = now.minute
    second = now.second + now.microsecond / 1000000

    hour_angle = (hour + minute / 60) * 30 - 90
    minute_angle = (minute + second / 60) * 6 - 90
    second_angle = second * 6 - 90

    hour_length = int(clock_radius * 0.45)
    hx = int(cx + hour_length * math.cos(math.radians(hour_angle)))
    hy = int(cy + hour_length * math.sin(math.radians(hour_angle)))
    cv2.line(frame, (cx, cy), (hx, hy), WHITE, 4, cv2.LINE_AA)

    minute_length = int(clock_radius * 0.65)
    mx = int(cx + minute_length * math.cos(math.radians(minute_angle)))
    my = int(cy + minute_length * math.sin(math.radians(minute_angle)))
    cv2.line(frame, (cx, cy), (mx, my), CYAN, 3, cv2.LINE_AA)

    second_length = int(clock_radius * 0.73)
    sx = int(cx + second_length * math.cos(math.radians(second_angle)))
    sy = int(cy + second_length * math.sin(math.radians(second_angle)))
    cv2.line(frame, (cx, cy), (sx, sy), RED, 1, cv2.LINE_AA)

    cv2.circle(frame, (cx, cy), 4, YELLOW, -1, cv2.LINE_AA)

    time_text = now.strftime("%I:%M:%S %p")

    text_size = cv2.getTextSize(time_text, cv2.FONT_HERSHEY_SIMPLEX, 0.32, 1)[0]

    text_x = cx - text_size[0] // 2
    text_y = cy + clock_radius + 18

    cv2.putText(frame, time_text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.32, CYAN, 1, cv2.LINE_AA)


# =========================================================
# SCREENSHOT
# =========================================================

def save_screenshot(frame):

    if not os.path.exists(SCREENSHOT_DIR):
        os.makedirs(SCREENSHOT_DIR)

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filepath = os.path.join(SCREENSHOT_DIR, f"holo_{timestamp}.png")

    cv2.imwrite(filepath, frame)

    return filepath


# =========================================================
# MAIN
# =========================================================

def main():

    # Let OpenCV use its internal optimizations and multiple threads
    # for its own operations (blur, resize, etc).
    cv2.setUseOptimized(True)
    try:
        cv2.setNumThreads(max(1, os.cpu_count() or 4))
    except Exception:
        pass

    ensure_model()
    detector = get_detector()

    # Plain default backend - most reliable across cameras/drivers.
    # (CAP_DSHOW / MJPG / BUFFERSIZE tweaks were removed because on many
    # webcam + driver combinations they cause cap.read() to hang or
    # freeze the whole window instead of speeding things up.)
    cap = cv2.VideoCapture(0)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

    if not cap.isOpened():
        print("Camera could not be opened.")
        detector.close()
        return

    particles = create_particles(PARTICLE_COUNT, FRAME_WIDTH, FRAME_HEIGHT)

    current_color = GREEN
    gesture_number = 0
    selected_mode = "PARTICLES"

    prev_time = time.time()
    fps_history = deque(maxlen=FPS_WINDOW)

    last_selection_time = 0
    animation_start = time.time()

    tracker = HandTracker()
    video_start_time = time.time()

    # --- Screenshot state ---
    screenshot_hold_start = None
    screenshot_cooldown_until = 0
    screenshot_flash_until = 0
    screenshot_flash_text = ""
    pending_screenshot = False

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)

    # ---------------------------------------------
    # PRE-ALLOCATED SCRATCH BUFFERS
    # Reused every frame instead of np.zeros_like(frame)
    # being allocated fresh each iteration - this was the
    # single biggest source of per-frame lag.
    # ---------------------------------------------
    glow_layer = np.zeros((FRAME_HEIGHT, FRAME_WIDTH, 3), dtype=np.uint8)
    line_layer = np.zeros_like(glow_layer)
    particle_layer = np.zeros_like(glow_layer)
    sphere_layer = np.zeros_like(glow_layer)
    panel_glow_scratch = np.zeros_like(glow_layer)

    try:
        while True:
            success, frame = cap.read()

            if not success:
                print("Camera frame error.")
                break

            frame = cv2.flip(frame, 1)
            height, width, _ = frame.shape

            # If the actual captured frame doesn't match our scratch
            # buffer size (camera gave us something different), resize
            # buffers once to match instead of silently breaking.
            if glow_layer.shape[0] != height or glow_layer.shape[1] != width:
                glow_layer = np.zeros((height, width, 3), dtype=np.uint8)
                line_layer = np.zeros_like(glow_layer)
                particle_layer = np.zeros_like(glow_layer)
                sphere_layer = np.zeros_like(glow_layer)
                panel_glow_scratch = np.zeros_like(glow_layer)

            animation_time = time.time() - animation_start

            # ---------------------------------------------
            # MEDIAPIPE DETECTION
            # camera_frame (the real webcam picture, with your face
            # etc.) is used ONLY for hand detection - it is never
            # shown on screen. Everything drawn/displayed from here
            # on uses a plain black canvas instead.
            # ---------------------------------------------
            camera_frame = frame
            rgb_frame = cv2.cvtColor(camera_frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

            timestamp_ms = int((time.time() - video_start_time) * 1000)
            result = detector.detect_for_video(mp_image, timestamp_ms)

            detected_hands = []
            if result.hand_landmarks:
                for hand in result.hand_landmarks:
                    detected_hands.append(landmarks_to_points(hand, width, height))

            all_hands = tracker.update(detected_hands)
            fresh_hands = tracker.fresh_hands(detected_hands, all_hands)

            # ---------------------------------------------
            # BLACK BACKGROUND
            # Replace the camera image with a plain black canvas so
            # only the hand skeleton / effects / UI are visible - no
            # face, no room, no live video.
            # ---------------------------------------------
            frame = np.zeros((height, width, 3), dtype=np.uint8)

            # ---------------------------------------------
            # LAYERS - clear the reused buffers instead of
            # allocating new ones.
            # ---------------------------------------------
            glow_layer.fill(0)
            line_layer.fill(0)
            particle_layer.fill(0)
            sphere_layer.fill(0)

            for hand_points in all_hands:
                draw_hand_skeleton(line_layer, hand_points)

            if len(all_hands) == 2:
                draw_dual_hand_web(glow_layer, line_layer, all_hands[0], all_hands[1])

            # ---------------------------------------------
            # GESTURE
            # ---------------------------------------------
            if fresh_hands:
                gesture_number = get_gesture(fresh_hands[0])

                if gesture_number == 1:
                    current_color = GREEN
                elif gesture_number == 2:
                    current_color = BLUE
                elif gesture_number == 3:
                    current_color = PURPLE
                elif gesture_number == 4:
                    current_color = RED
            else:
                gesture_number = 0

            # ---------------------------------------------
            # SCREENSHOT GESTURE — decide only, save at the end
            # ---------------------------------------------
            screenshot_progress = 0.0
            pending_screenshot = False

            if len(fresh_hands) == 2 and time.time() > screenshot_cooldown_until:

                g1 = get_gesture(fresh_hands[0])
                g2 = get_gesture(fresh_hands[1])

                palm_distance = math.hypot(
                    fresh_hands[0][0][0] - fresh_hands[1][0][0],
                    fresh_hands[0][0][1] - fresh_hands[1][0][1]
                )

                if g1 == 4 and g2 == 4 and palm_distance < SCREENSHOT_PALM_DISTANCE:

                    if screenshot_hold_start is None:
                        screenshot_hold_start = time.time()

                    held_for = time.time() - screenshot_hold_start
                    screenshot_progress = min(1.0, held_for / SCREENSHOT_HOLD_SECONDS)

                    if held_for >= SCREENSHOT_HOLD_SECONDS:
                        pending_screenshot = True
                        screenshot_flash_text = "SCREENSHOT SAVED"
                        screenshot_flash_until = time.time() + 2.0
                        screenshot_cooldown_until = time.time() + SCREENSHOT_COOLDOWN_SECONDS
                        screenshot_hold_start = None

                else:
                    screenshot_hold_start = None
            else:
                screenshot_hold_start = None

            # ---------------------------------------------
            # PARTICLES
            # ---------------------------------------------
            for particle in particles:
                particle.update(width, height, all_hands, gesture_number)

            draw_particles(particle_layer, particles, current_color)
            connect_particles(particle_layer, particles, current_color)

            # ---------------------------------------------
            # AURA
            # ---------------------------------------------
            if selected_mode == "AURA":
                for hand_points in all_hands:
                    draw_hand_aura(glow_layer, hand_points, current_color, animation_time)

            # ---------------------------------------------
            # HOLOGRAPHIC SPHERE + CLOCK
            # ---------------------------------------------
            sphere_distance = 0
            sphere_radius = 0

            # Sphere/circle only appears when CLOCK mode is explicitly
            # selected - bringing both hands together in any other mode
            # will not draw it.
            if len(all_hands) == 2 and selected_mode == "CLOCK":
                sphere_distance, sphere_radius = draw_holographic_sphere(
                    frame, sphere_layer, all_hands[0], all_hands[1], animation_time
                )

                center = (
                    int((all_hands[0][0][0] + all_hands[1][0][0]) / 2),
                    int((all_hands[0][0][1] + all_hands[1][0][1]) / 2)
                )

                draw_sphere_particles(sphere_layer, center, sphere_radius, animation_time)
                draw_holographic_clock(frame, center, sphere_radius)

            # ---------------------------------------------
            # PANEL / MENU SELECTION
            # ---------------------------------------------
            finger = None
            if fresh_hands:
                finger = fresh_hands[0][INDEX_TIP]

            hovered = draw_holo_panel(frame, finger, selected_mode, panel_glow_scratch)

            current_time = time.time()

            if (
                hovered is not None
                and gesture_number == 4
                and (current_time - last_selection_time > 0.8)
            ):
                selected_mode = hovered
                last_selection_time = current_time

                if selected_mode == "RESET":
                    for particle in particles:
                        particle.x = random.uniform(0, width)
                        particle.y = random.uniform(0, height)
                        particle.vx = random.uniform(-1, 1)
                        particle.vy = random.uniform(-1, 1)

                    selected_mode = "PARTICLES"

            # ---------------------------------------------
            # COMPOSITE EFFECTS (using fast_glow for speed)
            # ---------------------------------------------
            sphere_glow = fast_glow(sphere_layer, 18)
            frame = cv2.addWeighted(frame, 1.0, sphere_glow, 0.30, 0)

            particle_glow = fast_glow(particle_layer, 8)
            frame = cv2.addWeighted(frame, 1.0, particle_glow, 0.22, 0)

            glow_blur = fast_glow(glow_layer, 12)
            frame = cv2.addWeighted(frame, 1.0, glow_blur, 0.20, 0)

            frame = cv2.addWeighted(frame, 1.0, sphere_layer, 0.95, 0)
            frame = cv2.addWeighted(frame, 1.0, line_layer, 0.85, 0)
            frame = cv2.addWeighted(frame, 1.0, particle_layer, 0.90, 0)

            # ---------------------------------------------
            # TITLE
            # ---------------------------------------------
            cv2.putText(frame, "VIRTUAL AIR CANVAS", (20, 35),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75, CYAN, 2, cv2.LINE_AA)
            cv2.putText(frame, "AI HAND + HOLOGRAPHIC INTERACTION", (22, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.40, WHITE, 1, cv2.LINE_AA)

            # ---------------------------------------------
            # LIVE DASHBOARD (smoothed FPS)
            # ---------------------------------------------
            now = time.time()
            elapsed = now - prev_time
            prev_time = now

            instant_fps = 1 / elapsed if elapsed > 0 else 0
            fps_history.append(instant_fps)
            fps = sum(fps_history) / len(fps_history)

            gesture_names = {0: "FLOAT", 1: "ATTRACT", 2: "MAGNET", 3: "EXPLODE", 4: "PALM SELECT"}
            gesture_name = gesture_names.get(gesture_number, "FLOAT")

            dashboard_text = (
                f"HANDS: {len(all_hands)} | "
                f"GESTURE: {gesture_name} | "
                f"DISTANCE: {int(sphere_distance)}px | "
                f"FPS: {int(fps)}"
            )

            dash_size = cv2.getTextSize(dashboard_text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)[0]
            dash_x = (width - dash_size[0]) // 2
            dash_y = height - 70

            # Small ROI copy for the dashboard background instead of
            # copying the whole frame.
            dbg_x0 = max(0, dash_x - 16)
            dbg_y0 = max(0, dash_y - 26)
            dbg_x1 = min(width, dash_x + dash_size[0] + 16)
            dbg_y1 = min(height, dash_y + 10)

            dash_roi = frame[dbg_y0:dbg_y1, dbg_x0:dbg_x1]
            dash_bg = dash_roi.copy()
            dash_bg[:] = (20, 20, 20)
            dash_roi[:] = cv2.addWeighted(dash_roi, 0.55, dash_bg, 0.45, 0)

            cv2.rectangle(
                frame,
                (dash_x - 16, dash_y - 26),
                (dash_x + dash_size[0] + 16, dash_y + 10),
                CYAN, 1, cv2.LINE_AA
            )

            cv2.putText(frame, dashboard_text, (dash_x, dash_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, CYAN, 1, cv2.LINE_AA)

            cv2.putText(frame, f"MODE: {selected_mode}", (width - 250, 65),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.50, CYAN, 2, cv2.LINE_AA)

            if len(all_hands) == 2 and selected_mode == "CLOCK":
                cv2.putText(frame, f"SPHERE SIZE: {sphere_radius}", (width - 310, 95),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.43, PURPLE, 1, cv2.LINE_AA)

            # ---------------------------------------------
            # SCREENSHOT PROGRESS RING
            # ---------------------------------------------
            if screenshot_hold_start is not None and screenshot_progress > 0:

                ring_center = (width // 2, 150)
                ring_radius = 34

                cv2.circle(frame, ring_center, ring_radius, (60, 60, 60), 4, cv2.LINE_AA)

                end_angle = int(360 * screenshot_progress)
                cv2.ellipse(
                    frame, ring_center, (ring_radius, ring_radius),
                    -90, 0, end_angle, YELLOW, 4, cv2.LINE_AA
                )

                cv2.putText(frame, "HOLD FOR SCREENSHOT", (ring_center[0] - 110, ring_center[1] + 55),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, YELLOW, 1, cv2.LINE_AA)

            # ---------------------------------------------
            # BOTTOM INSTRUCTIONS
            # ---------------------------------------------
            cv2.putText(frame, "Both palms together = Screenshot",
                        (20, height - 45), cv2.FONT_HERSHEY_SIMPLEX, 0.43, WHITE, 1, cv2.LINE_AA)
            cv2.putText(frame, "1: Attract | 2: Magnet | 3: Explode | Open Palm: Select | Q: Quit",
                        (20, height - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.43, WHITE, 1, cv2.LINE_AA)

            # ---------------------------------------------
            # ACTUAL SCREENSHOT CAPTURE — frame is fully
            # composited at this point, so the saved PNG
            # matches exactly what's on screen.
            # ---------------------------------------------
            if pending_screenshot:
                saved_path = save_screenshot(frame)
                print(f"Screenshot saved: {saved_path}")

            if time.time() < screenshot_flash_until:
                cv2.putText(frame, screenshot_flash_text, (20, 95),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, GREEN, 2, cv2.LINE_AA)

            # ---------------------------------------------
            # DISPLAY
            # ---------------------------------------------
            cv2.imshow(WINDOW_NAME, frame)

            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break

            if cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()
        detector.close()


if __name__ == "__main__":
    main()