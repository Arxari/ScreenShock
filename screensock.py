import os
import time
import subprocess
import requests
import cv2
import numpy as np
from datetime import datetime, timedelta
import threading
import queue
from queue import Queue
from dataclasses import dataclass

# Config
REFERENCE_FOLDER = 'reference_images'
SCREENSHOT_FOLDER = 'screenshots'
API_URL = 'https://api.shocklink.net/2/shockers/control'
API_KEY = 'ApiKey'
SHOCK_ID = 'ShockerId'
SCREENSHOT_INTERVAL = 1 # in seconds how often to take a screenshot
DELETE_AFTER = timedelta(minutes=5) # after how long to detele old screenshots
SHOCK_COOLDOWN = 30 # how long to wait before shocking again if you get a repeated trigger/match
MATCH_THRESHOLD = 0.85 # how similar the screenshot must be to the reference image
VERBOSE = True

def log(message, level="INFO"):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] [{level}] {message}")

class OpenCLManager:
    def __init__(self):
        self.ocl_available = cv2.ocl.haveOpenCL()
        self.ocl_enabled = False
        self.failed_attempts = 0
        self.max_failures = 3
        self.last_reset = time.time()
        self.reset_interval = 300  # fail count cooldown reset

        if self.ocl_available:
            self.enable_opencl()

    def enable_opencl(self):
        try:
            cv2.ocl.setUseOpenCL(True)
            self.ocl_enabled = cv2.ocl.useOpenCL()
            if self.ocl_enabled:
                log("OpenCL acceleration enabled")
            else:
                log("Failed to enable OpenCL", "WARN")
        except Exception as e:
            log(f"Error enabling OpenCL: {e}", "ERROR")
            self.ocl_enabled = False

    def should_use_opencl(self):
        current_time = time.time()
        if current_time - self.last_reset > self.reset_interval:
            self.failed_attempts = 0
            self.last_reset = current_time

        return self.ocl_enabled and self.failed_attempts < self.max_failures

    def record_failure(self):
        self.failed_attempts += 1
        if self.failed_attempts >= self.max_failures:
            log("Too many OpenCL failures, switching to CPU", "WARN")
            self.ocl_enabled = False

ocl_manager = OpenCLManager()

def preprocess_template(image):
    if image is None:
        log("Invalid input image for preprocessing", "ERROR")
        return None

    try:
        processed = image.copy()
        if len(processed.shape) > 2:
            processed = cv2.cvtColor(processed, cv2.COLOR_BGR2GRAY)

        v = np.median(processed)
        lower = int(max(0, (1.0 - 0.33) * v))
        upper = int(min(255, (1.0 + 0.33) * v))

        if ocl_manager.should_use_opencl():
            try:
                edges = cv2.UMat(processed)
                edges = cv2.Canny(edges, lower, upper)
                return edges.get()
            except Exception as e:
                log(f"OpenCL Canny failed: {e}, falling back to CPU", "WARN")
                ocl_manager.record_failure()

        return cv2.Canny(processed, lower, upper)
    except Exception as e:
        log(f"Preprocessing error: {e}", "ERROR")
        return None

def match_template(screenshot, template):
    if screenshot is None or template is None:
        return 0

    try:
        if template.shape[0] > screenshot.shape[0] or template.shape[1] > screenshot.shape[1]:
            log("Template larger than screenshot, skipping match", "WARN")
            return 0

        if ocl_manager.should_use_opencl():
            try:
                src_umat = cv2.UMat(screenshot)
                tpl_umat = cv2.UMat(template)
                result = cv2.matchTemplate(src_umat, tpl_umat, cv2.TM_CCOEFF_NORMED)
                result = result.get()
            except Exception as e:
                log(f"OpenCL template matching failed: {e}", "WARN")
                ocl_manager.record_failure()
                result = cv2.matchTemplate(screenshot, template, cv2.TM_CCOEFF_NORMED)
        else:
            result = cv2.matchTemplate(screenshot, template, cv2.TM_CCOEFF_NORMED)

        _, max_val, _, _ = cv2.minMaxLoc(result)
        if VERBOSE:
            log(f"Match value: {max_val:.2f}")

        return max_val
    except Exception as e:
        log(f"Template matching failed: {e}", "ERROR")
        return 0

def load_reference_images():
    references = {}
    log("Loading reference images...")
    for filename in os.listdir(REFERENCE_FOLDER):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            try:
                path = os.path.join(REFERENCE_FOLDER, filename)
                img = cv2.imread(path)
                if img is None:
                    log(f"Failed to load {filename}", "ERROR")
                    continue

                processed = preprocess_template(img)
                if processed is not None:
                    references[filename] = processed
                    log(f"Loaded reference: {filename} ({processed.shape})")
                else:
                    log(f"Failed to process {filename}", "ERROR")
            except Exception as e:
                log(f"Error loading {filename}: {e}", "ERROR")
    log(f"Loaded {len(references)} reference images")
    return references

def detect_compositor():
    try:
        # KWin (KDE Plasma)
        kwin_check = subprocess.run(['ps', '-C', 'kwin_x11'], capture_output=True, text=True)
        if kwin_check.returncode == 0:
            return 'kwin'

        kwin_wayland_check = subprocess.run(['ps', '-C', 'kwin_wayland'], capture_output=True, text=True)
        if kwin_wayland_check.returncode == 0:
            return 'kwin'

        # Mutter (GNOME)
        mutter_check = subprocess.run(['ps', '-C', 'gnome-shell'], capture_output=True, text=True)
        if mutter_check.returncode == 0:
            return 'mutter'

        # Wlroots
        wayland_check = os.environ.get('WAYLAND_DISPLAY')
        if wayland_check:
            return 'wlroots'

        return 'unknown'
    except Exception as e:
        log(f"Compositor detection error: {e}", "ERROR")
        return 'unknown'

def capture_screen():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{SCREENSHOT_FOLDER}/screenshot_{timestamp}.png"

    compositor = detect_compositor()
    log(f"Detected compositor: {compositor}")

    screenshot_methods = {
        'kwin': ['spectacle', '-b', '-o'],
        'wlroots': ['grim', filename],
        'mutter': ['gnome-screenshot', '-f', filename],
        'unknown': ['import', '-window', 'root', filename]  # Fallback - ImageMagick
    }

    method = screenshot_methods.get(compositor, screenshot_methods['unknown'])

    for attempt in range(3):
        try:
            log(f"Capturing screenshot (attempt {attempt+1}): {filename}")

            if compositor in ['wlroots', 'unknown']:
                subprocess.run(method, check=True)
            else:
                full_method = method + [filename]
                subprocess.run(full_method, check=True)

            if not os.path.exists(filename):
                raise RuntimeError(f"Screenshot file not created: {filename}")

            log(f"Screenshot captured: {filename}")
            return filename
        except subprocess.CalledProcessError as e:
            log(f"Screenshot failed (attempt {attempt+1}): {e}", "ERROR")
            time.sleep(1)
        except Exception as e:
            log(f"Unexpected screenshot error: {e}", "ERROR")
            time.sleep(1)

    raise RuntimeError("Failed to capture screenshot after 3 attempts")

def send_api_request(screenshot_file, reference_file):
    try:
        headers = {
            'accept': 'application/json',
            'OpenShockToken': API_KEY,
            'Content-Type': 'application/json'
        }

        payload = {
            'shocks': [{
                'id': SHOCK_ID,
                'type': 'Shock',
                'intensity': 50,
                'duration': 1000,
                'exclusive': True
            }],
            'customName': 'ScreenShock'
        }

        start = time.time()
        log(f"Sending API request for {reference_file}...")
        response = requests.post(API_URL, headers=headers, json=payload, timeout=10)
        response.raise_for_status()

        elapsed = (time.time() - start) * 1000
        log(f"API success ({elapsed:.1f}ms): {reference_file}")
        return True
    except Exception as e:
        log(f"API failed: {e}", "ERROR")
        return False

def cleanup_old_screenshots():
    log("Starting screenshot cleanup...")
    now = datetime.now()
    deleted = 0
    for filename in os.listdir(SCREENSHOT_FOLDER):
        path = os.path.join(SCREENSHOT_FOLDER, filename)
        file_time = datetime.fromtimestamp(os.path.getctime(path))
        if (now - file_time) > DELETE_AFTER:
            try:
                os.remove(path)
                deleted += 1
                log(f"Deleted old screenshot: {filename}")
            except Exception as e:
                log(f"Cleanup failed: {filename} - {e}", "ERROR")
    log(f"Cleanup completed. Deleted {deleted} files")

@dataclass
class ScreenshotTask:
    path: str
    timestamp: float
    processed: bool = False

class ScreenShock:
    def __init__(self):
        self.screenshot_queue = Queue()
        self.reference_images = load_reference_images()
        self.last_shock_time = {}
        self.last_match_state = {}
        self.running = False

        if not self.reference_images:
            raise ValueError("No valid reference images loaded")

    def start(self):
        log("starting sc...")
        self.running = True
        threading.Thread(target=self._capture_loop, daemon=True).start()
        threading.Thread(target=self._process_loop, daemon=True).start()
        log("Processor started with 2 worker threads")

    def stop(self):
        log("stopping...")
        self.running = False

    def _capture_loop(self):
        log("Capture loop started")
        while self.running:
            try:
                start = time.time()
                path = capture_screen()
                self.screenshot_queue.put(ScreenshotTask(path, time.time()))
                sleep_time = max(0, SCREENSHOT_INTERVAL - (time.time() - start))
                if VERBOSE:
                    log(f"Capture loop sleeping for {sleep_time:.2f}s")
                time.sleep(sleep_time)
            except Exception as e:
                log(f"Capture error: {e}", "ERROR")
                time.sleep(SCREENSHOT_INTERVAL)

    def _process_loop(self):
        log("Processing loop started")
        while self.running:
            try:
                task = self.screenshot_queue.get(timeout=1)
                log(f"Processing screenshot: {os.path.basename(task.path)}")
                start_time = time.time()

                self._process_screenshot(task)

                elapsed = time.time() - start_time
                log(f"Finished processing {os.path.basename(task.path)} in {elapsed:.2f}s")
                self.screenshot_queue.task_done()
            except queue.Empty:
                pass
            except Exception as e:
                log(f"Processing error: {e}", "ERROR")

    def _process_screenshot(self, task):
        try:
            screenshot = cv2.imread(task.path, 0)  # grayscale
            if screenshot is None:
                log(f"Failed to read {task.path}", "ERROR")
                return

            edges = preprocess_template(screenshot)
            if edges is None:
                log(f"Failed to process {task.path}", "ERROR")
                return

            current_time = time.time()
            for ref_name, template in self.reference_images.items():
                match_val = match_template(edges, template)

                if match_val > MATCH_THRESHOLD:
                    last_time = self.last_shock_time.get(ref_name, 0)
                    time_since = current_time - last_time

                    log(f"MATCH FOUND: {ref_name} (confidence: {match_val:.2f})")

                    if time_since > SHOCK_COOLDOWN:
                        log(f"Triggering shock for {ref_name}")
                        if send_api_request(task.path, ref_name):
                            self.last_shock_time[ref_name] = current_time
                    else:
                        log(f"Cooldown active for {ref_name} ({SHOCK_COOLDOWN - time_since:.1f}s remaining)")
                else:
                    log(f"No match for {ref_name} (confidence: {match_val:.2f})")

            cleanup_old_screenshots()
        except Exception as e:
            log(f"Processing failed: {e}", "ERROR")

def main():
    os.makedirs(SCREENSHOT_FOLDER, exist_ok=True)
    log("starting...")

    try:
        processor = ScreenShock()
        processor.start()

        log("Running. Press Ctrl+C to exit.")
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        log("\nit's getting dark...")
        processor.stop()
    except Exception as e:
        log(f"Fatal error: {e}", "CRITICAL")
        raise

if __name__ == "__main__":
    main()
