import os
import time
import cv2
from detector import Detectorv3


def main(argc: int, *argv: str) -> int:
    base_dir = os.path.dirname(os.path.abspath(__file__))

    in_path = os.path.join(base_dir, "testvid.mp4")
    out_path = os.path.join(base_dir, "testvid_out.mp4")

    if not os.path.exists(in_path):
        print("video not found:", in_path)
        return 1

    det = Detectorv3(
        model_dir=r"C:\Users\dhair\Desktop\Code\RandomJawn\SquirrelDeter\src\models\squirrelnet_ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8\saved_model",
        squirrel_class_id=1,
        min_score=0.1,
    )

    cap = cv2.VideoCapture(in_path)
    if not cap.isOpened():
        print("failed to open video")
        return 1

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 1e-6:
        fps = 30.0

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_path, fourcc, fps, (width, height))
    if not writer.isOpened():
        print("failed to open video writer")
        cap.release()
        return 1

    processed = 0
    start_t = time.perf_counter()
    last_report_t = start_t
    last_report_frames = 0

    def print_progress(force: bool = False) -> None:
        nonlocal last_report_t, last_report_frames

        now = time.perf_counter()
        if not force and (now - last_report_t) < 0.5:
            return

        elapsed = now - start_t
        inst_frames = processed - last_report_frames
        inst_time = now - last_report_t
        proc_fps = (inst_frames / inst_time) if inst_time > 1e-6 else 0.0

        pct = (processed / total_frames * 100.0) if total_frames > 0 else 0.0
        avg_fps = (processed / elapsed) if elapsed > 1e-6 else 0.0

        if total_frames > 0 and avg_fps > 1e-6:
            eta_s = (total_frames - processed) / avg_fps
            eta_str = f"{eta_s:.1f}s"
        else:
            eta_str = "?"

        bar_len = 30
        filled = int(bar_len * (processed / total_frames)) if total_frames > 0 else 0
        bar = "=" * filled + "-" * (bar_len - filled)

        print(
            f"\r[{bar}] {pct:6.2f}%  {processed}/{total_frames if total_frames > 0 else '?'}  "
            f"proc_fps={proc_fps:5.2f}  avg_fps={avg_fps:5.2f}  ETA={eta_str}   ",
            end="",
            flush=True,
        )

        last_report_t = now
        last_report_frames = processed

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

        writer.write(out)

        processed += 1
        print_progress()

    print_progress(force=True)
    print()

    cap.release()
    writer.release()

    print("saved:", out_path)
    return 0


if __name__ == "__main__":
    argv = __import__("sys").argv
    exit(main(len(argv), *argv))
