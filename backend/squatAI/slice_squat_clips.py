import pandas as pd
import os
import subprocess

# Ścieżka do ffmpeg.exe
ffmpeg_path = r"C:\ProgramData\chocolatey\bin\ffmpeg.exe"

# Ścieżka do folderu, w którym znajduje się skrypt
script_dir = os.path.dirname(os.path.abspath(__file__))

# Ścieżki
video_folder = os.path.join(script_dir, '..', 'competition_videos')
output_folder = os.path.join(script_dir, 'squat_clips')
timestamps_file = os.path.join(script_dir, 'squat_timestamps.csv')
os.makedirs(output_folder, exist_ok=True)

# Debugowanie
print("Script directory:", script_dir)
print("Video folder:", video_folder)
print("Looking for file at:", timestamps_file)
if not os.path.exists(timestamps_file):
    print("File does NOT exist at the specified path!")
    print("Current working directory:", os.getcwd())
    print("Files in script directory:", os.listdir(script_dir))
    exit()

# Wczytaj timestamps
df = pd.read_csv(timestamps_file)
print("CSV loaded successfully:", df.head())

# Konwersja czasu na sekundy
def to_seconds(time_str):
    if ':' in time_str:
        parts = time_str.split(':')
        if len(parts) == 3:  # HH:MM:SS format
            hours, minutes, seconds = map(int, parts)
            return hours * 3600 + minutes * 60 + seconds
        elif len(parts) == 2:  # MM:SS format
            minutes, seconds = map(int, parts)
            return minutes * 60 + seconds
        else:
            raise ValueError(f"Invalid time format: {time_str}")
    return float(time_str)

# Wycinanie klipów
for idx, row in df.iterrows():
    video_path = os.path.join(video_folder, row['video_file'])
    start_time = to_seconds(row['start_time'])
    end_time = to_seconds(row['end_time'])
    label = row['label']

    print(f"Processing clip {idx + 1}: start={start_time}s, end={end_time}s, video={video_path}")

    # Sprawdź, czy plik wideo istnieje
    if not os.path.exists(video_path):
        print(f"Video file not found: {video_path}")
        continue

    # Wytnij klip za pomocą FFmpeg
    try:
        output_file = os.path.join(output_folder, f'squat_{idx + 1}_label_{label}_vid3.mp4')
        duration = end_time - start_time
        cmd = [
            ffmpeg_path,
            "-i", video_path,
            "-ss", str(start_time),
            "-t", str(duration),
            "-c:v", "libx264",
            "-c:a", "aac",  # Dodano kodek audio (jeśli wideo zawiera audio)
            "-b:v", "1M",   # Bitrate wideo
            "-an" if label == "0" else "-c:a", "aac",  # Wyłącz audio dla label=0, zachowaj dla label=1
            "-y",           # Nadpisz plik, jeśli istnieje
            output_file
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0:
            print(f"Saved clip: {output_file}")
        else:
            print(f"Error processing clip {idx + 1}: {result.stderr}")
    except Exception as e:
        print(f"Error processing clip {idx + 1}: {e}")

print("Zapisano klipów:", len(os.listdir(output_folder)))