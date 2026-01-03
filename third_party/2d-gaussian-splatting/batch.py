import subprocess
from pathlib import Path

VIDEO_DIR = Path("G:/drive-download-20260102T212517Z-1-001")          # folder containing .mp4 files
OUTPUT_ROOT = Path("G:/drive-download-20260102T212517Z-1-001")        # where xxx/input, xxx/outputs live
FPS = 10
ITERATIONS = 15000

GS_DIR = Path("third_party/2d-gaussian-splatting")

def run(cmd, cwd=None):
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, cwd=cwd, check=True)

def process_video(video_path: Path):
    name = video_path.stem
    scene_dir = OUTPUT_ROOT / name
    input_dir = scene_dir / "input"
    input_dir.mkdir(parents=True, exist_ok=True)

    # 1. Extract frames
    run([
        "python", "extract_frames.py",
        "--video_path", str(video_path.resolve()),
        "--output_dir", str(input_dir.resolve()),
        "--target_fps", str(FPS)
    ], cwd=GS_DIR)

    # 2. Run SfM
    run([
        "python", "convert.py",
        "-s", str(scene_dir.resolve())
    ], cwd=GS_DIR)

    # 3. Train 2DGS
    run([
        "python", "train.py",
        "-s", str(scene_dir.resolve()),
        "--iteration", str(ITERATIONS)
    ], cwd=GS_DIR)

    # 4. Extract mesh
    run([
        "python", "render.py",
        "-s", str(scene_dir.resolve()),
        "-m", str(scene_dir.resolve()),
        "--skip_test",
        "--skip_train",
        "--mesh_res", "1024",
        "--depth_trunc", "5"
    ], cwd=GS_DIR)

def main():
    videos = sorted(VIDEO_DIR.glob("*.mp4"))
    if not videos:
        raise RuntimeError(f"No mp4 files found in {VIDEO_DIR}")

    for video in videos:
        print(f"\n=== Processing {video.name} ===")
        if (OUTPUT_ROOT / video.stem / "train" / "ours_14500" / "fuse_decimated.ply").exists():
            print(f"Skipping {video.name}, output already exists.")
            continue
        process_video(video)

if __name__ == "__main__":
    main()
