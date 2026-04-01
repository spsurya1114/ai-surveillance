
import os
from ingestion.video_processor import process_video

base_path = "data/videos"

for cam in os.listdir(base_path):
    cam_path = os.path.join(base_path, cam)

    if os.path.isdir(cam_path):
        for video in os.listdir(cam_path):
            video_path = os.path.join(cam_path, video)

            print(f"Processing {video_path}...")
            process_video(video_path, cam)

print("Done.")

