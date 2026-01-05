import subprocess
import os
import tempfile
import shutil

def rotate_mp4_ccw_90_inplace(mp4_path):
    assert mp4_path.endswith(".mp4")
    
    tmp_fd, tmp_path = tempfile.mkstemp(suffix=".mp4")
    os.close(tmp_fd)

    # transpose=2 表示逆时针 90°
    cmd = [
        "ffmpeg",
        "-y",
        "-i", mp4_path,
        "-vf", "transpose=2",
        "-c:a", "copy",   # 音频不重新编码
        tmp_path
    ]

    subprocess.run(cmd, check=True)

    shutil.move(tmp_path, mp4_path)

# example
rotate_mp4_ccw_90_inplace(r"G:\drive-download-20260103T224221Z-1-001/VID_20260103_150607.mp4")