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
     """
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
     parser.add_argument("--video_path", type=str, help="Path to the input video file.")
     parser.add_argument("--output_dir", type=str, help="Directory to save extracted frames.")
     parser.add_argument("--target_fps", type=float, default=1.0, help="Target frames per second.")
     parser.add_argument("--img_ext", type=str, default="png", help="Image file extension (png/jpg).")
     parser.add_argument("--start_frame", type=int, default=0, help="Frame index to start extraction from.")
     parser.add_argument("--max_frames", type=int, default=None, help="Maximum number of frames to extract.")
     args = parser.parse_args()

     extract_frames(
          video_path=args.video_path,
          output_dir=args.output_dir,
          target_fps=args.target_fps,
          img_ext=args.img_ext,
          start_frame=args.start_frame,
          max_frames=args.max_frames,
     )