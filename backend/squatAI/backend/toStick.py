import cv2
import numpy as np
import os
import mediapipe as mp
import csv
import subprocess
import json

mp_pose = mp.solutions.pose
POSE_CONNECTIONS = mp_pose.POSE_CONNECTIONS
JOINT_NAMES = [l.name for l in mp_pose.PoseLandmark]
landmark_positions = {}


def calculate_body(landmarks):
    global landmark_positions
    if landmarks is None or not isinstance(landmarks, (list, np.ndarray)) or len(landmarks) < 33:
        landmark_positions.clear()
        return
    try:
        lm = np.array([[lm[0], lm[1]] for lm in landmarks])
        landmark_positions['left_shoulder'] = lm[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
        landmark_positions['right_shoulder'] = lm[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
        landmark_positions['left_hip'] = lm[mp_pose.PoseLandmark.LEFT_HIP.value]
        landmark_positions['right_hip'] = lm[mp_pose.PoseLandmark.RIGHT_HIP.value]
        landmark_positions['left_ankle'] = lm[mp_pose.PoseLandmark.LEFT_ANKLE.value]
        landmark_positions['right_ankle'] = lm[mp_pose.PoseLandmark.RIGHT_ANKLE.value]
        landmark_positions['left_knee'] = lm[mp_pose.PoseLandmark.LEFT_KNEE.value]
        landmark_positions['right_knee'] = lm[mp_pose.PoseLandmark.RIGHT_KNEE.value]
        landmark_positions['nose'] = lm[mp_pose.PoseLandmark.NOSE.value]
        landmark_positions['left_heel'] = lm[mp_pose.PoseLandmark.LEFT_HEEL.value]
        landmark_positions['right_heel'] = lm[mp_pose.PoseLandmark.RIGHT_HEEL.value]
        landmark_positions['left_foot_index'] = lm[mp_pose.PoseLandmark.LEFT_FOOT_INDEX.value]
        landmark_positions['right_foot_index'] = lm[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX.value]
    except Exception as e:
        print(f"Error calculating body landmarks: {e}")
        landmark_positions.clear()

def calculate_camera_angles(landmarks):
    if not landmark_positions:
        return None, None

    try:
        shoulder_width = np.linalg.norm(landmark_positions['right_shoulder'] - landmark_positions['left_shoulder'])
        hip_width = np.linalg.norm(landmark_positions['right_hip'] - landmark_positions['left_hip'])
        shoulder_center = (landmark_positions['left_shoulder'] + landmark_positions['right_shoulder']) / 2
        hip_center = (landmark_positions['left_hip'] + landmark_positions['right_hip']) / 2
        ankle_center = (landmark_positions['left_ankle'] + landmark_positions['right_ankle']) / 2
        knee_center = (landmark_positions['left_knee'] + landmark_positions['right_knee']) / 2
        left_torso_width = np.linalg.norm(landmark_positions['left_shoulder'] - landmark_positions['left_hip'])
        right_torso_width = np.linalg.norm(landmark_positions['right_shoulder'] - landmark_positions['right_hip'])

        torso_sum = right_torso_width + left_torso_width
        if torso_sum == 0: torso_sum = 0.001

        torso_asymmetry = (right_torso_width - left_torso_width) / torso_sum

        if shoulder_width == 0: shoulder_width = 0.001
        if hip_width == 0: hip_width = 0.001

        shoulder_hip_offset = (shoulder_center[0] - hip_center[0]) / max(shoulder_width, hip_width)
        horizontal_angle = torso_asymmetry * 45 + shoulder_hip_offset * 30
        torso_vector = hip_center - shoulder_center
        torso_angle = np.arctan2(torso_vector[0], torso_vector[1]) * 180 / np.pi
        upper_leg_vector = knee_center - hip_center
        upper_leg_angle = np.arctan2(upper_leg_vector[0], upper_leg_vector[1]) * 180 / np.pi
        lower_leg_vector = ankle_center - knee_center
        lower_leg_angle = np.arctan2(lower_leg_vector[0], lower_leg_vector[1]) * 180 / np.pi
        head_to_foot_vector = ankle_center - np.linalg.norm(landmark_positions['nose'])
        overall_body_angle = np.arctan2(head_to_foot_vector[0], head_to_foot_vector[1]) * 180 / np.pi
        ankle_width = np.linalg.norm(landmark_positions['right_ankle'] - landmark_positions['left_ankle'])

        width_ratio = hip_width / shoulder_width
        width_deviation = abs(width_ratio - 1.0)
        ankle_shoulder_ratio = ankle_width / shoulder_width
        ankle_deviation = abs(ankle_shoulder_ratio - 0.6)

        vertical_angle = overall_body_angle * 0.7

        vertical_angle += torso_angle * 0.3

        if width_deviation > 0.3:
            vertical_angle += (width_deviation - 0.3) * 20

        return horizontal_angle, vertical_angle

    except Exception as e:
        print(f"Error calculating camera angles: {e}")
        return None, None

def smooth_landmarks(history, current):
    if current is None:
        return None
    history.append(current)
    if len(history) > 5:
        history.pop(0)
    avg = np.mean([h for h in history if h is not None], axis=0)
    return avg


def calculate_person_height(landmarks):
    if not landmark_positions:
        return None

    try:
        foot_points = [
            landmark_positions['left_heel'],
            landmark_positions['right_heel'],
            landmark_positions['left_foot_index'],
            landmark_positions['right_foot_index']
        ]
        lowest_foot_y = max([point[1] for point in foot_points])
        person_height = lowest_foot_y - landmark_positions['nose'][1]
        return max(person_height, 0)

    except Exception as e:
        print(f"Error calculating height: {e}")
        return None


def calculate_segment_lengths(landmarks):

    if landmarks is None:
        return None, None, None, None

    try:
        left_thigh = np.linalg.norm(landmark_positions['left_knee'] - landmark_positions['left_hip'])
        right_thigh = np.linalg.norm(landmark_positions['right_knee'] - landmark_positions['right_hip'])
        left_shin = np.linalg.norm(landmark_positions['left_ankle'] - landmark_positions['left_knee'])
        right_shin = np.linalg.norm(landmark_positions['right_ankle'] - landmark_positions['right_knee'])

        return left_thigh, right_thigh, left_shin, right_shin

    except Exception as e:
        print(f"Error calculating segment lengths: {e}")
        return None, None, None, None


def calculate_height_features(person_height):
    features = {}

    if not landmark_positions or person_height is None or person_height <= 0:
        return {key: None for key in [
            'left_hip_height_percent', 'right_hip_height_percent',
            'left_knee_height_percent', 'right_knee_height_percent',
            'left_shoulder_height_percent', 'right_shoulder_height_percent',
            'left_hip_height', 'right_hip_height',
            'left_knee_height', 'right_knee_height',
            'left_shoulder_height', 'right_shoulder_height'
        ]}

    try:
        foot_points = [
            landmark_positions['left_heel'],
            landmark_positions['right_heel'],
            landmark_positions['left_foot_index'],
            landmark_positions['right_foot_index']
        ]
        lowest_foot_y = max([point[1] for point in foot_points])

        left_hip_height = lowest_foot_y - landmark_positions['left_hip'][1]
        right_hip_height = lowest_foot_y - landmark_positions['right_hip'][1]
        left_knee_height = lowest_foot_y - landmark_positions['left_knee'][1]
        right_knee_height = lowest_foot_y - landmark_positions['right_knee'][1]
        left_shoulder_height = lowest_foot_y - landmark_positions['left_shoulder'][1]
        right_shoulder_height = lowest_foot_y - landmark_positions['right_shoulder'][1]

        features['left_hip_height_percent'] = (left_hip_height / person_height) * 100
        features['right_hip_height_percent'] = (right_hip_height / person_height) * 100
        features['left_knee_height_percent'] = (left_knee_height / person_height) * 100
        features['right_knee_height_percent'] = (right_knee_height / person_height) * 100
        features['left_shoulder_height_percent'] = (left_shoulder_height / person_height) * 100
        features['right_shoulder_height_percent'] = (right_shoulder_height / person_height) * 100

        features['left_hip_height'] = left_hip_height
        features['right_hip_height'] = right_hip_height
        features['left_knee_height'] = left_knee_height
        features['right_knee_height'] = right_knee_height
        features['left_shoulder_height'] = left_shoulder_height
        features['right_shoulder_height'] = right_shoulder_height

    except Exception as e:
        print(f"Error calculating height features: {e}")
        return {key: None for key in features.keys()}

    return features


def export_height_analysis_csv(squat_features, person_height, thigh_length, shin_length, horizontal_angle,
                               vertical_angle, frame_times, out_path):

    with open(out_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)

        headers = [
            'left_hip_height_percent', 'right_hip_height_percent',
            'left_knee_height_percent', 'right_knee_height_percent',
            'left_shoulder_height_percent', 'right_shoulder_height_percent',
            'person_height', 'thigh_length', 'shin_length',
            'horizontal_camera_angle', 'vertical_camera_angle', 'frame_time_seconds'
        ]
        writer.writerow(headers)

        for frame_idx, frame_features in enumerate(squat_features):
            row = []

            for header in headers[:-6]:
                row.append(frame_features.get(header, None))

            row.extend([person_height, thigh_length, shin_length, horizontal_angle, vertical_angle])

            if frame_idx < len(frame_times):
                row.append(frame_times[frame_idx])
            else:
                row.append(None)

            writer.writerow(row)

def interpolate_missing_landmarks(all_landmarks):
    for i in range(len(all_landmarks)):
        if all_landmarks[i] is None:
            prev_idx = i - 1
            while prev_idx >= 0 and all_landmarks[prev_idx] is None:
                prev_idx -= 1
            next_idx = i + 1
            while next_idx < len(all_landmarks) and all_landmarks[next_idx] is None:
                next_idx += 1
            if prev_idx >= 0 and next_idx < len(all_landmarks):
                prev_landmarks = np.array(all_landmarks[prev_idx])
                next_landmarks = np.array(all_landmarks[next_idx])
                ratio = (i - prev_idx) / (next_idx - prev_idx)
                interp = prev_landmarks + (next_landmarks - prev_landmarks) * ratio
                all_landmarks[i] = interp.tolist()
            elif prev_idx >= 0:
                all_landmarks[i] = all_landmarks[prev_idx]
            elif next_idx < len(all_landmarks):
                all_landmarks[i] = all_landmarks[next_idx]
    return all_landmarks

def apply_temporal_smoothing(all_landmarks, window_size=5):
    if len(all_landmarks) < window_size:
        return all_landmarks

    smoothed_landmarks = []

    for i in range(len(all_landmarks)):
        if all_landmarks[i] is None:
            smoothed_landmarks.append(None)
            continue

        start_idx = max(0, i - window_size // 2)
        end_idx = min(len(all_landmarks), i + window_size // 2 + 1)

        valid_landmarks = []
        for j in range(start_idx, end_idx):
            if all_landmarks[j] is not None:
                valid_landmarks.append(np.array(all_landmarks[j]))

        if valid_landmarks:
            avg_landmarks = np.mean(valid_landmarks, axis=0)
            smoothed_landmarks.append(avg_landmarks.tolist())
        else:
            smoothed_landmarks.append(all_landmarks[i])

    return smoothed_landmarks


def processing(video_path, output_path, pose):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open input video {video_path}")
        return [], []

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    rotation = 0
    try:
        result = subprocess.run(['ffprobe', '-i', video_path, '-show_streams', '-v', 'quiet', '-print_format', 'json'],
                                capture_output=True, text=True, check=False)
        probe_data = json.loads(result.stdout)
        if 'streams' in probe_data and len(probe_data['streams']) > 0:
            for stream in probe_data['streams']:
                if stream.get('codec_type') == 'video':
                    side_data = stream.get('side_data_list', [])
                    for data in side_data:
                        if 'rotation' in data:
                            rotation = float(data['rotation'])
                            print(f"  - Detected input rotation: {rotation}°")
                            break
                    break
    except subprocess.CalledProcessError as e:
        print(f"Warning: ffprobe failed to retrieve metadata: {e}")
    except json.JSONDecodeError:
        print(f"Warning: Could not parse ffprobe output for {video_path}")

    if abs(rotation) % 180 == 90:
        out_width, out_height = height, width
    else:
        out_width, out_height = width, height

    fourcc = cv2.VideoWriter_fourcc(*'avc1')
    out = cv2.VideoWriter(output_path, fourcc, fps, (out_width, out_height))

    if not out.isOpened():
        print(f"Error: Could not create output video {output_path}")
        cap.release()
        return [], []

    all_landmarks = []
    frame_times = []
    frame_count = 0
    print("  - Extracting pose landmarks...")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1

        if rotation != 0:
            if rotation == 90:
                frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
                width, height = height, width
            elif rotation == 180:
                frame = cv2.rotate(frame, cv2.ROTATE_180)
            elif rotation == 270 or rotation == -90:
                frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
                width, height = height, width

        current_time_sec = frame_count / fps
        frame_times.append(current_time_sec)
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(image_rgb)

        if results.pose_landmarks:
            pixel_landmarks = []
            for lm in results.pose_landmarks.landmark:
                x_px = int(lm.x * width)
                y_px = int(lm.y * height)
                pixel_landmarks.append([x_px, y_px, lm.z, lm.visibility])
            all_landmarks.append(pixel_landmarks)
            for connection in POSE_CONNECTIONS:
                start_idx = connection[0]
                end_idx = connection[1]
                pt1 = tuple(np.array(pixel_landmarks[start_idx][:2], dtype=int))
                pt2 = tuple(np.array(pixel_landmarks[end_idx][:2], dtype=int))
                cv2.line(frame, pt1, pt2, (0, 255, 0), 2)
        else:
            all_landmarks.append(None)
            print(f"  - No person detected in frame {frame_count}")

        out.write(frame)
        if frame_count % 100 == 0:
            print(f"  - Processed {frame_count} frames")

    cap.release()
    out.release()
    return all_landmarks, frame_times


def process_videos(video_dir, output_dir):
    pose = mp_pose.Pose(static_image_mode=False, model_complexity=2, enable_segmentation=False,
                        min_detection_confidence=0.5, min_tracking_confidence=0.5)

    if not os.path.isdir(video_dir) or not os.listdir(video_dir):
        return

    for file_name in os.listdir(video_dir):
        video_extensions = ('.mp4', '.mov', '.avi', '.mkv', '.MP4', '.MOV', '.AVI', '.MKV')
        if not file_name.lower().endswith(tuple(ext.lower() for ext in video_extensions)):
            continue

        print(f"Processing: {file_name}")
        video_path = os.path.join(video_dir, file_name)

        output_path = os.path.join(output_dir, f"{os.path.splitext(file_name)[0]}_stick.mp4")
        trajectory_path = os.path.join(output_dir, f"{os.path.splitext(file_name)[0]}_height_analysis.csv")

        all_landmarks, frame_times = processing(video_path, output_path, pose)

        if not all_landmarks:
            continue

        none_count = sum(1 for x in all_landmarks if x is None)
        print(f"  - Missing frames before interpolation: {none_count}/{len(all_landmarks)}")

        print("  - Interpolating missing frames...")
        all_landmarks = interpolate_missing_landmarks(all_landmarks)

        print("  - Applying temporal smoothing...")
        all_landmarks = apply_temporal_smoothing(all_landmarks, window_size=5)

        print("  - Calculating person height and segment lengths...")
        heights = []
        thigh_lengths = []
        shin_lengths = []
        horizontal_angles = []
        vertical_angles = []

        for i in range(min(10, len(all_landmarks))):
            if all_landmarks[i] is not None:
                calculate_body(all_landmarks[i])
                person_height = calculate_person_height(all_landmarks[i])
                if person_height is not None:
                    heights.append(person_height)

                left_thigh, right_thigh, left_shin, right_shin = calculate_segment_lengths(all_landmarks[i])
                if all([x is not None for x in [left_thigh, right_thigh, left_shin, right_shin]]):
                    thigh_lengths.append((left_thigh + right_thigh) / 2)
                    shin_lengths.append((left_shin + right_shin) / 2)
                h_angle, v_angle = calculate_camera_angles(all_landmarks[i])
                if h_angle is not None and v_angle is not None:
                    horizontal_angles.append(h_angle)
                    vertical_angles.append(v_angle)

        person_height = np.median(heights) if heights else None
        thigh_length = np.median(thigh_lengths) if thigh_lengths else None
        shin_length = np.median(shin_lengths) if shin_lengths else None
        horizontal_angle = np.median(horizontal_angles) if horizontal_angles else 0.0
        vertical_angle = np.median(vertical_angles) if vertical_angles else 0.0

        if person_height is None:
            print(f"  - WARNING: Could not calculate person height for {file_name}")

        print(f"  - Person height: {person_height:.1f} pixels")
        print(
            f"  - Thigh length: {thigh_length:.1f} pixels" if thigh_length else "  - Could not calculate thigh length")
        print(f"  - Shin length: {shin_length:.1f} pixels" if shin_length else "  - Could not calculate shin length")

        print("  - Interpolating missing frames...")
        all_landmarks = interpolate_missing_landmarks(all_landmarks)
        print("  - Applying temporal smoothing...")
        all_landmarks = apply_temporal_smoothing(all_landmarks, window_size=5)

        if person_height is None and all_landmarks:
            for frame_landmarks in all_landmarks[:10]:
                if frame_landmarks is not None:
                    calculate_body(frame_landmarks)
                    person_height = calculate_person_height(frame_landmarks)
                    if person_height is not None:
                        heights.append(person_height)
            person_height = np.median(heights) if heights else None

        if person_height is None:
            print(f"  - WARNING: Could not calculate person height for {file_name}")
            continue
        else:
            print("  - Calculating height-based features...")
            all_squat_features = []
            for frame_idx, frame_landmarks in enumerate(all_landmarks):
                if frame_landmarks is not None:
                    calculate_body(frame_landmarks)
                height_features = calculate_height_features(person_height)
                all_squat_features.append(height_features)

        export_height_analysis_csv(all_squat_features, person_height, thigh_length, shin_length,
                                   horizontal_angle, vertical_angle, frame_times, trajectory_path)

        print(f"Completed: {file_name}")
        print(f"  - Stick figure video: {output_path}")
        print(f"  - Height analysis CSV: {trajectory_path}")

def ToCSV(InputDir, OutputDir):
    video_dir = InputDir
    output_dir = OutputDir

    os.makedirs(output_dir, exist_ok=True)

    process_videos(video_dir, output_dir)
    return output_dir