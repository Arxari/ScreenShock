import os
import time
import subprocess
import requests
from PIL import Image
import pytesseract
import cv2
import numpy as np
from datetime import datetime, timedelta

REFERENCE_FOLDER = 'reference_images'
SCREENSHOT_FOLDER = 'screenshots'
API_URL = 'https://api.shocklink.net/2/shockers/control'
API_KEY = 'apikey'
SHOCK_ID = 'shockerid'
SCREENSHOT_INTERVAL = 5  # seconds
DELETE_AFTER = timedelta(minutes=5)
SHOCK_COOLDOWN = 60  # seconds
MATCH_THRESHOLD = 0.8  # threshold for template matching

def capture_screen():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{SCREENSHOT_FOLDER}/screenshot_{timestamp}.png"
    subprocess.run(["grim", filename], check=True)
    return filename

def perform_ocr(image):
    return pytesseract.image_to_string(image)

def load_reference_images():
    references = {}
    for filename in os.listdir(REFERENCE_FOLDER):
        if filename.endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(REFERENCE_FOLDER, filename)
            image = cv2.imread(image_path)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            text = perform_ocr(Image.fromarray(gray))
            references[filename] = {'image': gray, 'text': text.strip()}
    return references

def send_api_request():
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
        'customName': 'ScreenS'
    }

    response = requests.post(url=API_URL, headers=headers, json=payload)

    if response.status_code == 200:
        print('API request sent successfully.')
    else:
        print(f"Failed to send API request. Response: {response.content}")

def cleanup_old_screenshots():
    now = datetime.now()
    for filename in os.listdir(SCREENSHOT_FOLDER):
        file_path = os.path.join(SCREENSHOT_FOLDER, filename)
        file_creation_time = datetime.fromtimestamp(os.path.getctime(file_path))
        if now - file_creation_time > DELETE_AFTER:
            os.remove(file_path)
            print(f"Deleted old screenshot: {filename}")

def text_similarity(text1, text2):
    set1 = set(text1.lower().split())
    set2 = set(text2.lower().split())
    return len(set1.intersection(set2)) / len(set1.union(set2)) if set1 or set2 else 0

def match_template(screenshot, template):
    result = cv2.matchTemplate(screenshot, template, cv2.TM_CCOEFF_NORMED)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
    return max_val

def main():
    os.makedirs(SCREENSHOT_FOLDER, exist_ok=True)
    reference_images = load_reference_images()
    print(f"Loaded {len(reference_images)} reference images.")

    last_shock_time = {}
    last_match_state = {}

    while True:
        screenshot_path = capture_screen()
        screenshot = cv2.imread(screenshot_path, 0)  # reads it as grayscale, this might be changed but it works rn sooooo
        screenshot_text = perform_ocr(Image.fromarray(screenshot))

        print(f"\nScreenshot taken: {screenshot_path}")
        print(f"Screenshot OCR text: {screenshot_text[:100]}...")

        current_time = time.time()

        for filename, ref_data in reference_images.items():
            match_value = match_template(screenshot, ref_data['image'])
            text_sim = text_similarity(screenshot_text, ref_data['text'])

            print(f"\nChecking {filename}:")
            print(f"  Template match value: {match_value:.2f}")
            print(f"  Text similarity: {text_sim:.2f}")

            is_match = (match_value > MATCH_THRESHOLD or text_sim > 0.7)

            if is_match and (filename not in last_match_state or not last_match_state[filename]):
                print(f"Match found for {filename}!")
                if filename not in last_shock_time or current_time - last_shock_time[filename] > SHOCK_COOLDOWN:
                    send_api_request()
                    last_shock_time[filename] = current_time
                else:
                    print(f"Cooldown active for {filename}. Skipping shock.")
            elif is_match:
                print(f"Continuous match for {filename}. No action taken.")
            else:
                print(f"No match for {filename}.")

            last_match_state[filename] = is_match

        cleanup_old_screenshots()
        time.sleep(SCREENSHOT_INTERVAL)

if __name__ == "__main__":
    main()
