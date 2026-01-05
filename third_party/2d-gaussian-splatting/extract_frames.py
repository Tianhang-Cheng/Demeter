import cv2
import os
import math

def extract_frames(
    video_path: str,
    output_dir: str,
    target_fps: float,
    img_ext: str = "png",
    start_frame: int = 0,
    max_frames: int | None = None,
    scale_ratio: float = 1.0,
):
    """
    Extract frames from a video at a given FPS.

    Args:
        video_path: path to .mp4
        output_dir: directory to save frames
        target_fps: desired FPS (e.g. 1, 5, 10)
        img_ext: image format (png / jpg)
        start_frame: frame index to start from
        max_frames: maximum number of frames to save
        scale_ratio: resize ratio (e.g. 0.5 -> half size)
    """
    assert scale_ratio > 0, "scale_ratio must be > 0"

    os.makedirs(output_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    assert cap.isOpened(), f"Failed to open {video_path}"

    video_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    step = max(1, int(round(video_fps / target_fps)))

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    frame_idx = start_frame
    saved = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if (frame_idx - start_frame) % step == 0:
            if scale_ratio != 1.0:
                h, w = frame.shape[:2]
                new_w = int(w * scale_ratio)
                new_h = int(h * scale_ratio)
                frame = cv2.resize(
                    frame, (new_w, new_h), interpolation=cv2.INTER_AREA
                )

            out_path = os.path.join(output_dir, f"{saved:06d}.{img_ext}")
            cv2.imwrite(out_path, frame)
            saved += 1

            if max_frames is not None and saved >= max_frames:
                break

        frame_idx += 1

    cap.release()
    print(f"Saved {saved} frames to {output_dir}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Extract frames from video.")
    parser.add_argument("--video_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--target_fps", type=float, default=1.0)
    parser.add_argument("--img_ext", type=str, default="png")
    parser.add_argument("--start_frame", type=int, default=0)
    parser.add_argument("--max_frames", type=int, default=None)
    parser.add_argument("--scale_ratio", type=float, default=1.0,
                        help="Resize ratio (e.g. 0.5 for half resolution)")

    args = parser.parse_args()

    extract_frames(
        video_path=args.video_path,
        output_dir=args.output_dir,
        target_fps=args.target_fps,
        img_ext=args.img_ext,
        start_frame=args.start_frame,
        max_frames=args.max_frames,
        scale_ratio=args.scale_ratio,
    )
