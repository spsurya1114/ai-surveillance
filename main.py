
from ingestion.video_processor import process_video

if __name__ == "__main__":
    videos = [
        ("data/videos/cam1.mp4", "CAM_01"),
        ("data/videos/cam2.mp4", "CAM_02"),
        ("data/videos/cam3.mp4", "CAM_03")
    ]

    for video_path, cam_id in videos:
        print(f"Processing {cam_id}...")
        process_video(video_path, cam_id)

