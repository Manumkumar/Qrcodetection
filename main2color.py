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
from threading import Thread, Lock
import queue

# --- CONFIGURABLE CONSTANTS ---
CAMERA_INDEX = 0  # Default camera index
CAMERA_NAME = "eMeet s600"
API_URL = "https://cbofskoflvqdsvfgbfyj.supabase.co/functions/v1/qr-api"
FRAME_WIDTH = 1920
FRAME_HEIGHT = 1080
API_TIMEOUT = 10

# --- Performance Optimizations ---
PROCESS_EVERY_N_FRAMES = 2
DISPLAY_SCALE = 0.8
MAX_QUEUE_SIZE = 5

# --- Auto Zoom Parameters ---
TARGET_QR_SIZE = 400
ZOOM_SMOOTHING = 0.1
ZOOM_MIN = 1.0
ZOOM_MAX = 2.0
SEARCH_ZOOM_STEP = 0.25
SEARCH_HOLD_TIME = 3.0

# --- Snapshot Parameters ---
SNAPSHOT_FOLDER = "snapshots"
SNAPSHOT_INTERVAL = 10

# --- Deduplication and Verification ---
DUPLICATE_FORGET_TIME = 10.0
VERIFICATION_SCANS = 2

# --- Batch Parameters ---
BATCH_SIZE = 6

# --- LOGGING SETUP ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)

os.environ['PYZBAR_WARNINGS'] = '0'

def digital_zoom(frame, zoom_factor, center=None):
    if zoom_factor <= 1.0:
        return frame
    h, w = frame.shape[:2]
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
    zoomed = cv2.resize(cropped, (w, h), interpolation=cv2.INTER_LINEAR)
    return zoomed

def preprocess_image_fast(frame):
    if len(frame.shape) == 3:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    else:
        gray = frame
    return gray

def preprocess_image_thorough(frame):
    if len(frame.shape) == 3:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    else:
        gray = frame
    gray = cv2.equalizeHist(gray)
    gray = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 11, 2
    )
    kernel = np.array([[-1, -1, -1],
                       [-1,  9, -1],
                       [-1, -1, -1]])
    sharp = cv2.filter2D(gray, -1, kernel)
    return sharp

class QRDetector:
    def __init__(self):
        self.detected_codes = {}
        self.candidate_codes = {}
        self.sent_codes = set()
        self.batch = []
        self.lock = Lock()
        self.api_queue = queue.Queue(maxsize=100)
        self.api_thread = Thread(target=self._api_worker, daemon=True)
        self.api_thread.start()

    def _api_worker(self):
        while True:
            try:
                batch_data = self.api_queue.get(timeout=1)
                if batch_data is None:
                    break
                self._send_to_api(batch_data)
                self.api_queue.task_done()
            except queue.Empty:
                continue
            except Exception as e:
                logging.error(f"API worker error: {e}")

    def _send_to_api(self, batch_data):
        try:
            response = requests.post(
                API_URL,
                json={"qr_data_list": batch_data},
                timeout=API_TIMEOUT
            )
            if response.ok:
                logging.info(f"✅ Sent batch of {len(batch_data)} QR codes")
                return True
            else:
                logging.error(f"❌ Failed batch: {response.status_code}")
                return False
        except Exception as e:
            logging.error(f"API Error: {str(e)}")
            return False

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

    def detect_qr_codes(self, frame, thorough=False):
        with self.suppress_warnings():
            if thorough:
                processed = preprocess_image_thorough(frame)
            else:
                processed = preprocess_image_fast(frame)
            codes = pyzbar.decode(processed, symbols=[pyzbar.ZBarSymbol.QRCODE])
            if not codes and not thorough:
                return codes
            if not codes and thorough:
                codes = pyzbar.decode(frame, symbols=[pyzbar.ZBarSymbol.QRCODE])
            with self.lock:
                current_time = time.time()
                for code in codes:
                    try:
                        data = code.data.decode('utf-8')
                        if data in self.sent_codes:
                            continue
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
                                self.add_to_batch(data)
                                self.sent_codes.add(data)
                                del self.candidate_codes[data]
                    except Exception as e:
                        logging.warning(f"Error processing QR code: {e}")
                        continue
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
        if self.batch:
            try:
                self.api_queue.put_nowait(self.batch.copy())
                self.batch.clear()
                return True
            except queue.Full:
                logging.warning("API queue full, dropping batch")
                return False
        return False

def initialize_camera():
    cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)
    if not cap.isOpened():
        logging.error("Camera could not be opened.")
        sys.exit(1)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, 30)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
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
    process_frame_count = 0
    fps = 0.0
    process_fps = 0.0
    zoom_factor = 0.5
    search_mode = True
    search_zoom = ZOOM_MIN
    search_direction = 1
    search_start_time = time.time()
    last_qr_time = 0
    FOCUS_HOLD_TIME = 5.0
    QR_LOST_TIMEOUT = 2.0
    os.makedirs(SNAPSHOT_FOLDER, exist_ok=True)
    last_snapshot_time = time.time()
    snapshot_count = 0
    display_width = int(FRAME_WIDTH * DISPLAY_SCALE)
    display_height = int(FRAME_HEIGHT * DISPLAY_SCALE)
    window_name = f"{CAMERA_NAME} - QR Scanner (Optimized)"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, display_width, display_height)

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                logging.warning("Camera disconnected - reinitializing...")
                cap.release()
                time.sleep(1)
                cap = initialize_camera()
                continue
            frame_count += 1
            current_time = time.time()
            should_process = (frame_count % PROCESS_EVERY_N_FRAMES) == 0
            codes = []
            if should_process:
                process_frame_count += 1
                codes = detector.detect_qr_codes(frame, thorough=False)
            zoom_center = None
            if codes:
                search_mode = False
                last_qr_time = current_time
                largest_code = max(codes, key=lambda c: c.rect.width * c.rect.height)
                qr_x, qr_y, qr_w, qr_h = largest_code.rect
                zoom_center = (qr_x + qr_w // 2, qr_y + qr_h // 2)
                zoom_factor = min(ZOOM_MAX, max(ZOOM_MIN, TARGET_QR_SIZE / max(qr_w, 1)))
            else:
                if (current_time - last_qr_time) > QR_LOST_TIMEOUT:
                    search_mode = True
                    zoom_factor = ZOOM_MIN
                    zoom_center = None
                else:
                    search_mode = False
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
            codes_zoomed = detector.detect_qr_codes(frame_zoomed, thorough=should_process)
            for code in codes_zoomed:
                (x, y, w, h) = code.rect
                try:
                    data = code.data.decode('utf-8')
                    if data in detector.sent_codes:
                        color = (255, 0, 0)
                        status = "SENT"
                    else:
                        color = (0, 255, 0)
                        status = "NEW"
                    cv2.rectangle(frame_zoomed, (x, y), (x + w, y + h), color, 3)
                    corner_length = 20
                    cv2.line(frame_zoomed, (x, y), (x + corner_length, y), color, 4)
                    cv2.line(frame_zoomed, (x, y), (x, y + corner_length), color, 4)
                    cv2.line(frame_zoomed, (x + w, y), (x + w - corner_length, y), color, 4)
                    cv2.line(frame_zoomed, (x + w, y), (x + w, y + corner_length), color, 4)
                    cv2.line(frame_zoomed, (x, y + h), (x + corner_length, y + h), color, 4)
                    cv2.line(frame_zoomed, (x, y + h), (x, y + h - corner_length), color, 4)
                    cv2.line(frame_zoomed, (x + w, y + h), (x + w - corner_length, y + h), color, 4)
                    cv2.line(frame_zoomed, (x + w, y + h), (x + w, y + h - corner_length), color, 4)
                    text_lines = []
                    if len(data) > 30:
                        text_lines.append(f"{data[:27]}...")
                    else:
                        text_lines.append(data)
                    text_lines.append(f"[{status}]")
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    font_scale = 0.6
                    thickness = 2
                    max_text_width = 0
                    total_text_height = 0
                    line_height = 25
                    for line in text_lines:
                        (text_width, text_height), _ = cv2.getTextSize(line, font, font_scale, thickness)
                        max_text_width = max(max_text_width, text_width)
                        total_text_height += line_height
                    text_x = x
                    text_y = y - total_text_height - 10
                    if text_y < 0:
                        text_y = y + h + 10
                    if text_x + max_text_width > frame_zoomed.shape[1]:
                        text_x = frame_zoomed.shape[1] - max_text_width - 10
                    overlay = frame_zoomed.copy()
                    cv2.rectangle(overlay,
                                (text_x - 5, text_y - 5),
                                (text_x + max_text_width + 10, text_y + total_text_height + 5),
                                (0, 0, 0), -1)
                    cv2.addWeighted(overlay, 0.7, frame_zoomed, 0.3, 0, frame_zoomed)
                    for i, line in enumerate(text_lines):
                        cv2.putText(frame_zoomed, line,
                                  (text_x, text_y + (i + 1) * line_height),
                                  font, font_scale, color, thickness)
                except Exception as e:
                    cv2.rectangle(frame_zoomed, (x, y), (x + w, y + h), (0, 0, 255), 2)
                    cv2.putText(frame_zoomed, "DECODE ERROR", (x, y - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            if codes_zoomed and (current_time - last_snapshot_time >= SNAPSHOT_INTERVAL):
                snapshot_filename = os.path.join(SNAPSHOT_FOLDER, f"snapshot_{snapshot_count}.jpg")
                cv2.imwrite(snapshot_filename, frame_zoomed)
                logging.info(f"Snapshot saved: {snapshot_filename}")
                last_snapshot_time = current_time
                snapshot_count += 1
                try:
                    snapshot_img = cv2.imread(snapshot_filename)
                    if snapshot_img is not None:
                        codes_from_snapshot = detector.detect_qr_codes(snapshot_img, thorough=True)
                        if codes_from_snapshot:
                            logging.info(f"Processed snapshot: {len(codes_from_snapshot)} QR code(s)")
                except Exception as e:
                    logging.error(f"Error processing snapshot: {e}")
            if frame_count >= 30:
                now = time.time()
                fps = frame_count / (now - prev_time)
                process_fps = process_frame_count / (now - prev_time)
                prev_time = now
                frame_count = 0
                process_frame_count = 0
            cv2.putText(frame_zoomed, f"Display FPS: {fps:.1f}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.putText(frame_zoomed, f"Process FPS: {process_fps:.1f}", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            with detector.lock:
                total_detected = len(detector.detected_codes)
                total_sent = len(detector.sent_codes)
                total_candidates = len(detector.candidate_codes)
                batch_size = len(detector.batch)
            cv2.putText(frame_zoomed, f"Detected: {total_detected} | Sent: {total_sent}", (10, 150),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(frame_zoomed, f"Candidates: {total_candidates} | Batch: {batch_size}", (10, 175),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(frame_zoomed, f"Zoom: {zoom_factor:.2f}", (10, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
            if search_mode:
                cv2.putText(frame_zoomed, f"Search: {search_zoom:.2f}", (10, 120),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            legend_y = frame_zoomed.shape[0] - 70
            cv2.putText(frame_zoomed, "QR Status:", (10, legend_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(frame_zoomed, "BLUE: Already sent to API", (10, legend_y + 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
            cv2.putText(frame_zoomed, "GREEN: New detection", (10, legend_y + 45),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            frame_display = cv2.resize(frame_zoomed, (display_width, display_height))
            cv2.imshow(window_name, frame_display)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == ord('Q'):
                break
    except KeyboardInterrupt:
        logging.info("Interrupted by user.")
    finally:
        detector.api_queue.put(None)
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        main()

