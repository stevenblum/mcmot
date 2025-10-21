import cv2
import time
import os

# Directory to save snapshots
os.makedirs("camera_snapshots", exist_ok=True)

print("Scanning for available cameras...\n")

caps = []
window_names = []
for i in range(10):
    cap = cv2.VideoCapture(i)
    if not cap.isOpened():
        print(f"❌ Camera index {i}: not available")
        continue
    ret, frame = cap.read()
    if not ret or frame is None:
        print(f"⚠️ Camera index {i}: opened but no frame captured")
        cap.release()
        continue
    caps.append(cap)
    window_names.append(f"Camera {i}")
    print(f"✅ Camera index {i}: ready")

if not caps:
    print("No cameras available.")
    exit(0)

print("Press 's' to save snapshots, 'q' to quit.")

while True:
    for idx, cap in enumerate(caps):
        ret, frame = cap.read()
        if ret and frame is not None:
            cv2.imshow(window_names[idx], frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('s'):
        for idx, cap in enumerate(caps):
            ret, frame = cap.read()
            if ret and frame is not None:
                filename = f"camera_snapshots/camera_{idx}.jpg"
                cv2.imwrite(filename, frame)
                print(f"Snapshot saved as {filename}")
    if key == ord('q'):
        break

for cap in caps:
    cap.release()
cv2.destroyAllWindows()
print("\nDone! Snapshots are in ./camera_snapshots/")
