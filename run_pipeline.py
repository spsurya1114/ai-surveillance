import os
from pathlib import Path
from ingestion.video_processor import process_video

def main():
    base_path = Path("data/videos")
    
    # 1. Automatically create directories if missing
    base_path.mkdir(parents=True, exist_ok=True)
    expected_cams = ["CAM_01", "CAM_02", "CAM_03"]
    for cam in expected_cams:
        (base_path / cam).mkdir(parents=True, exist_ok=True)

    # Find active camera folders
    cameras = [d for d in base_path.iterdir() if d.is_dir()]
    cameras_found = len(cameras)
    
    # Filter only supported video extensions
    video_extensions = {".mp4", ".avi", ".mov", ".mkv"}
    videos_to_process = []

    for cam_dir in cameras:
        cam_name = cam_dir.name
        for item in cam_dir.iterdir():
            if item.is_file() and item.suffix.lower() in video_extensions:
                videos_to_process.append((item, cam_name))

    videos_found = len(videos_to_process)

    if videos_found == 0:
        print("\n==================================================")
        print("🎥 AI SURVEILLANCE PIPELINE SUMMARY")
        print("==================================================")
        print("No surveillance videos found in 'data/videos/' camera subfolders.")
        print("Please place your video files (.mp4, .avi, .mov, .mkv) in:")
        print(f"  - {base_path / 'CAM_01'}")
        print(f"  - {base_path / 'CAM_02'}")
        print(f"  - {base_path / 'CAM_03'}")
        print("==================================================")
        print("Exiting gracefully. No processing performed.")
        return

    print(f"\nProcessing {videos_found} videos across {cameras_found} camera directories...")

    videos_success = 0
    videos_failed = 0

    for video_path, cam_name in videos_to_process:
        print(f"\nProcessing {video_path.as_posix()} for {cam_name}...")
        try:
            # Continue calling process_video(video_path, cam) as a string path
            process_video(str(video_path), cam_name)
            videos_success += 1
        except Exception as e:
            print(f"❌ Error processing {video_path.as_posix()} for {cam_name}: {e}")
            videos_failed += 1

    print("\n==================================================")
    print("🎥 AI SURVEILLANCE PIPELINE PROCESSING SUMMARY")
    print("==================================================")
    print(f"Cameras found: {cameras_found}")
    print(f"Videos found: {videos_found}")
    print(f"Videos processed successfully: {videos_success}")
    print(f"Videos failed: {videos_failed}")
    print("==================================================")

if __name__ == "__main__":
    main()
