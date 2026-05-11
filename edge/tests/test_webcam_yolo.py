"""Quick offline smoke test - run YOLO on the PC's webcam.
Usage: python -m edge.tests.test_webcam_yolo
"""

import sys
import cv2

sys.path.insert(0, __file__.rsplit("edge", 1)[0] + "edge")

from vision.detector import YoloDetector  # noqa: E402
from vision.spatial import zh_label  # noqa: E402


def main():
    det = YoloDetector(weights="yolov8n.pt", device="cpu", conf=0.4)
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise SystemExit("No webcam found.")
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            for d in det.infer(frame):
                cv2.rectangle(frame, (d.x1, d.y1), (d.x2, d.y2), (0, 255, 0), 2)
                cv2.putText(frame, f"{zh_label(d.label)} {d.conf:.2f}",
                            (d.x1, max(20, d.y1 - 5)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
            cv2.imshow("yolo-webcam", frame)
            if cv2.waitKey(1) & 0xFF in (ord('q'), 27):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
