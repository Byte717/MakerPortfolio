import cv2
from detector import Detectorv3


def main(argc: int, *argv: str) -> int:
    det = Detectorv3(
        model_dir=r"C:\Users\dhair\Desktop\Code\RandomJawn\SquirrelDeter\src\models\squirrelnet_ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8\saved_model",
        squirrel_class_id=1,
        min_score=0.5,
    )

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("failed to open webcam")
        return 1

    DISPLAY_SCALE = 0.5 

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        squirrel_present, detections = det.forward(frame)

        out = frame.copy()
        for d in detections:
            if d["class_id"] != 1:
                continue
            x, y, w, h = d["box"]
            cv2.rectangle(out, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(
                out,
                f"squirrel {d['score']:.2f}",
                (x, max(0, y - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )

        if squirrel_present:
            cv2.putText(
                out,
                "SQUIRREL!",
                (15, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 255, 0),
                3,
                cv2.LINE_AA,
            )

        out_small = cv2.resize(
            out,
            None,
            fx=DISPLAY_SCALE,
            fy=DISPLAY_SCALE,
            interpolation=cv2.INTER_LINEAR,
        )

        cv2.imshow("detector", out_small)

        if cv2.waitKey(1) & 0xFF == 27: 
            break

    cap.release()
    cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    import sys
    exit(main(len(sys.argv), *sys.argv))
