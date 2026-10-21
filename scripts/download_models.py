"""Download the small face models (YuNet + SFace) from the OpenCV model zoo into ./models."""
import os
import sys
import urllib.request

BASE = "https://github.com/opencv/opencv_zoo/raw/main/models"
FILES = {
    "models/face_detection_yunet_2023mar.onnx": f"{BASE}/face_detection_yunet/face_detection_yunet_2023mar.onnx",
    "models/face_recognition_sface_2021dec.onnx": f"{BASE}/face_recognition_sface/face_recognition_sface_2021dec.onnx",
}

root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ok = True
for rel, url in FILES.items():
    dest = os.path.join(root, rel)
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    if os.path.exists(dest) and os.path.getsize(dest) > 100_000:
        print("already have", rel)
        continue
    print("downloading", url)
    try:
        urllib.request.urlretrieve(url, dest)
        print(f"  -> {rel} ({os.path.getsize(dest) / 1e6:.1f} MB)")
        if os.path.getsize(dest) < 100_000:
            print("  !! file too small (probably a Git-LFS pointer). Download it manually from the opencv_zoo repo.")
            ok = False
    except Exception as e:
        print("  failed:", e)
        ok = False
print("Done. Now set face.enabled: true in config.yaml and enrol photos with scripts/enroll_face.py" if ok else "Some downloads failed.")
sys.exit(0 if ok else 1)
