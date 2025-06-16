import time
import pyautogui

print("Mouse Jiggler is running...")

while True:
    # Move mouse slightly every 5 minutes
    pyautogui.move(1, 0)
    time.sleep(10)
    pyautogui.move(-1, 0)
    time.sleep(290)  # Wait for remainder of 5 minutes