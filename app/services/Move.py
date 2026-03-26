import pyautogui
import time
import sys
 
print("Press Ctrl+C to stop.")
 
try:
    while True:
        # Get current mouse position
        x, y = pyautogui.position()
       
        # Move mouse slightly (10 pixels right, then back)
        pyautogui.moveRel(10, 0, duration=0.2)
        pyautogui.moveRel(-10, 0, duration=0.2)
       
        # Press a harmless key (like Shift) just in case
        pyautogui.press('shift')
       
        print(f"Movement simulated at {time.strftime('%H:%M:%S')}")
       
        # Wait for 60 seconds before next movement
        time.sleep(60)
 
except KeyboardInterrupt:
    print("\nStopped.")