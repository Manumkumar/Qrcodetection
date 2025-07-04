import cv2
from pyzbar import pyzbar
import requests
import time
import warnings
import os
import sys
import logging
from contextlib import contextmanager
import datetime
import numpy as np

# --- CONFIGURABLE CONSTANTS ---
CAMERA_INDEX = 0
CAMERA_NAME = "eMeet s600"
API_URL = "https://cbofskoflvqdsvfgbfyj.supabase.co/functions/v1/qr-api"
FRAME_WIDTH = 3840
FRAME_HEIGHT = 2160
API_TIMEOUT = 10

# --- Auto Zoom Parameters ---
TARGET_QR_SIZE = 300
ZOOM_SMOOTHING = 0.1
ZOOM_MIN = 0.90
ZOOM_MAX = 3.0
SEARCH_ZOOM_STEP = 0.05
SEARCH_HOLD_TIME = 2.0

# --- Snapshot Parameters ---
SNAPSHOT_FOLDER = "snapshots"
SNAPSHOT_INTERVAL = 15  # seconds

# --- Deduplication and Verification ---
DUPLICATE_FORGET_TIME = 10.0  # seconds
VERIFICATION_SCANS = 2  # frames

# --- Batch Parameters ---
BATCH_SIZE = 6  # Number of QR codes to send at once

# --- LOGGING SETUP ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)

os.environ['PYZBAR_WARNINGS'] = '0'

def digital_zoom(frame, zoom_factor, center=None):
    h, w = frame.shape[:2]
    if zoom_factor <= 1.0:
        return frame
    if center is None:
        center_x, center_y = w // 2, h // 2
    else:
        center_x, center_y = center
    new_w = int(w / zoom_factor)
    new_h = int(h / zoom_factor)
    x1 = max(0, center_x - new_w // 2)
    y1 = max(0, center_y - new_h // 2)
    x2 = min(w, x1 + new_w)
    y2 = min(h, y1 + new_h)
    if x2 - x1 < new_w:
        x1 = max(0, w - new_w)
        x2 = w
    if y2 - y1 < new_h:
        y1 = max(0, h - new_h)
        y2 = h
    cropped = frame[y1:y2, x1:x2]
    return cv2.resize(cropped, (w, h), interpolation=cv2.INTER_CUBIC)

def preprocess_image(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    gray = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 11, 2
    )
    kernel = np.array([[-1, -1, -1],
                       [-1,  9, -1],
                       [-1, -1, -1]])
    sharp = cv2.filter2D(gray, -1, kernel)
    cv2.imshow("enhanced", sharp)
    return sharp

class QRDetector:
    def __init__(self):
        self.detected_codes = {}      # {data: last_detection_time}
        self.candidate_codes = {}     # {data: [first_time, confirm_count]}
        self.sent_codes = set()       # Track sent QR codes
        self.batch = []               # Batch list for QR data

    @contextmanager
    def suppress_warnings(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            original_stderr = sys.stderr
            sys.stderr = open(os.devnull, 'w')
            try:
                yield
            finally:
                sys.stderr.close()
                sys.stderr = original_stderr

    def detect_qr_codes(self, frame):
        with self.suppress_warnings():
            codes = pyzbar.decode(frame, symbols=[pyzbar.ZBarSymbol.QRCODE])
            if not codes:
                processed = preprocess_image(frame)
                codes = pyzbar.decode(processed, symbols=[pyzbar.ZBarSymbol.QRCODE])
            current_time = time.time()
            for code in codes:
                data = code.data.decode('utf-8')
                if data in self.sent_codes:
                    continue  # Already sent, skip
                last_time = self.detected_codes.get(data, 0)
                if current_time - last_time < DUPLICATE_FORGET_TIME:
                    continue
                if data not in self.candidate_codes:
                    self.candidate_codes[data] = [current_time, 1]
                else:
                    self.candidate_codes[data][1] += 1
                    if self.candidate_codes[data][1] >= VERIFICATION_SCANS:
                        self.detected_codes[data] = current_time
                        logging.info(f"👍 Verified QR: {data}")
                        self.add_to_batch(data)  # Add to batch instead of sending immediately
                        self.sent_codes.add(data)
                        del self.candidate_codes[data]
            # Clean up old candidates
            to_remove = []
            for data, (first_time, count) in self.candidate_codes.items():
                if current_time - first_time > 2 * DUPLICATE_FORGET_TIME:
                    to_remove.append(data)
            for data in to_remove:
                del self.candidate_codes[data]
            return codes

    def add_to_batch(self, data):
        self.batch.append({
            "qr_content": data,
            "scanner_id": CAMERA_NAME,
            "timestamp": datetime.datetime.now().isoformat()
        })
        if len(self.batch) >= BATCH_SIZE:
            self.send_batch_to_api()

    def send_batch_to_api(self):
        try:
            response = requests.post(
                API_URL,
                json={"qr_data_list": self.batch},  # Send list of QR data
                timeout=API_TIMEOUT
            )
            if response.ok:
                logging.info(f"✅ Sent batch of {len(self.batch)} QR codes")
                self.batch.clear()  # Clear batch after successful send
                return True
            else:
                logging.error(f"❌ Failed batch: {response.status_code}")
                return False
        except Exception as e:
            logging.error(f"API Error: {str(e)}")
            return False

def initialize_camera():
    cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)
    if not cap.isOpened():
        logging.error("Camera could not be opened.")
        sys.exit(1)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
    actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    logging.info(f"Camera resolution: {actual_width}x{actual_height}")
    cap.set(cv2.CAP_PROP_AUTOFOCUS, 1)
    return cap

def main():
    detector = QRDetector()
    cap = initialize_camera()
    prev_time = time.time()
    frame_count = 0
    zoom_factor = 0.5
    search_mode = True
    search_zoom = ZOOM_MIN
    search_direction = 1
    search_start_time = time.time()

    last_qr_time = 0
    FOCUS_HOLD_TIME = 5.0

    os.makedirs(SNAPSHOT_FOLDER, exist_ok=True)
    last_snapshot_time = time.time()
    snapshot_count = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                logging.warning("Camera disconnected - reinitializing...")
                cap.release()
                time.sleep(1)
                cap = initialize_camera()
                continue

            codes = detector.detect_qr_codes(frame)
            current_time = time.time()

            zoom_center = None
            if codes:
                search_mode = False
                last_qr_time = current_time
                largest_code = max(codes, key=lambda c: c.rect.width * c.rect.height)
                qr_x, qr_y, qr_w, qr_h = largest_code.rect
                zoom_center = (qr_x + qr_w // 2, qr_y + qr_h // 2)
            else:
                if (current_time - last_qr_time) < FOCUS_HOLD_TIME:
                    search_mode = False
                else:
                    search_mode = True

            if search_mode:
                zoom_factor = search_zoom
                zoom_center = None
                if current_time - search_start_time >= SEARCH_HOLD_TIME:
                    search_zoom += search_direction * SEARCH_ZOOM_STEP
                    if search_zoom >= ZOOM_MAX:
                        search_zoom = ZOOM_MAX
                        search_direction = -1
                    elif search_zoom <= ZOOM_MIN:
                        search_zoom = ZOOM_MIN
                        search_direction = 1
                    search_start_time = current_time

            frame_zoomed = digital_zoom(frame, zoom_factor, center=zoom_center)
            codes_zoomed = detector.detect_qr_codes(frame_zoomed)
            for code in codes_zoomed:
                (x, y, w, h) = code.rect
                cv2.rectangle(frame_zoomed, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(frame_zoomed, code.data.decode(), (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # --- SNAPSHOT LOGIC (save zoomed frame) ---
            if codes and (current_time - last_snapshot_time >= SNAPSHOT_INTERVAL):
                snapshot_filename = os.path.join(SNAPSHOT_FOLDER, f"snapshot_{snapshot_count}.jpg")
                cv2.imwrite(snapshot_filename, frame_zoomed)  # Save the current displayed zoomed frame
                logging.info(f"Snapshot saved: {snapshot_filename}")
                last_snapshot_time = current_time
                snapshot_count += 1

            frame_count += 1
            if frame_count >= 10:
                now = time.time()
                fps = frame_count / (now - prev_time)
                prev_time = now
                frame_count = 0
                cv2.putText(frame_zoomed, f"FPS: {fps:.2f}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
            cv2.putText(frame_zoomed, f"Zoom: {zoom_factor:.2f}", (10, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
            if search_mode:
                cv2.putText(frame_zoomed, f"Search: {search_zoom:.2f} ({search_direction})", (10, 110),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                hold_left = SEARCH_HOLD_TIME - (current_time - search_start_time)
                cv2.putText(frame_zoomed, f"Hold: {max(0, hold_left):.1f}s", (10, 150),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            cv2.imshow(f"{CAMERA_NAME} - QR Scanner", frame_zoomed)

            key = cv2.waitKey(1)
            if key == ord('q') or key == ord('Q'):
                break

    except KeyboardInterrupt:
        logging.info("Interrupted by user.")
    finally:
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        main()
