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

landmark_data = {}


def calculate_body_side_view(landmarks, width, height, visibility_threshold=0.0):

    global landmark_data
    landmark_data.clear()

    if landmarks is None or not isinstance(landmarks, (list, np.ndarray)) or len(landmarks) < 33:
        return None, None

    try:
        is_normalized = all(len(lm) == 4 for lm in landmarks)

        for i, lm in enumerate(landmarks):
            landmark_name = JOINT_NAMES[i]

            if is_normalized:
                x_px = float(lm[0]) * float(width)
                y_px = float(lm[1]) * float(height)
                visibility = float(lm[3])
            else:
                x_px = float(lm[0])
                y_px = float(lm[1])
                visibility = float(lm[3]) if len(lm) == 4 else 0.0

            landmark_data[landmark_name] = np.array([x_px, y_px, visibility], dtype=np.float64)

        left_critical_joints = [
            mp_pose.PoseLandmark.LEFT_HIP.name, mp_pose.PoseLandmark.LEFT_KNEE.name,
            mp_pose.PoseLandmark.LEFT_ANKLE.name, mp_pose.PoseLandmark.LEFT_FOOT_INDEX.name
        ]
        right_critical_joints = [
            mp_pose.PoseLandmark.RIGHT_HIP.name, mp_pose.PoseLandmark.RIGHT_KNEE.name,
            mp_pose.PoseLandmark.RIGHT_ANKLE.name, mp_pose.PoseLandmark.RIGHT_FOOT_INDEX.name
        ]

        left_visibility_sum = sum(landmark_data[name][2] for name in left_critical_joints)
        right_visibility_sum = sum(landmark_data[name][2] for name in right_critical_joints)

        main_side = 'LEFT' if left_visibility_sum >= right_visibility_sum else 'RIGHT'

        main_joints = [f'{main_side}_HIP', f'{main_side}_KNEE', f'{main_side}_ANKLE', f'{main_side}_SHOULDER']
        if any(landmark_data[name][2] < visibility_threshold for name in main_joints):
            return None, None

        return main_side, landmark_data

    except Exception:
        landmark_data.clear()
        return None, None


def calculate_segment_lengths_and_height_factor(current_landmark_data):

    if not current_landmark_data:
        return None
    try:
        left_joints = ['LEFT_HIP', 'LEFT_KNEE', 'LEFT_ANKLE']
        right_joints = ['RIGHT_HIP', 'RIGHT_KNEE', 'RIGHT_ANKLE']

        left_visibility_sum = sum(
            current_landmark_data[joint][2] for joint in left_joints if joint in current_landmark_data)
        right_visibility_sum = sum(
            current_landmark_data[joint][2] for joint in right_joints if joint in current_landmark_data)

        main_side = 'LEFT' if left_visibility_sum >= right_visibility_sum else 'RIGHT'

        main_shoulder_coord = current_landmark_data[f'{main_side}_SHOULDER'][:2]
        main_hip_coord = current_landmark_data[f'{main_side}_HIP'][:2]
        main_knee_coord = current_landmark_data[f'{main_side}_KNEE'][:2]
        main_ankle_coord = current_landmark_data[f'{main_side}_ANKLE'][:2]

        torso_length = np.linalg.norm(main_shoulder_coord - main_hip_coord)
        thigh_length = np.linalg.norm(main_hip_coord - main_knee_coord)
        shin_length = np.linalg.norm(main_knee_coord - main_ankle_coord)

        if torso_length < 10 or thigh_length < 10 or shin_length < 10:
            return None

        segment_sum = torso_length + thigh_length + shin_length

        estimated_full_height = segment_sum / 0.87

        return (estimated_full_height, torso_length, thigh_length, shin_length)

    except KeyError:
        return None
    except Exception:
        return None


def calculate_angle_3pt(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)

    ba = a - b
    bc = c - b

    magnitude_product = np.linalg.norm(ba) * np.linalg.norm(bc)

    if magnitude_product == 0:
        return 0.0

    cosine_angle = np.dot(ba, bc) / magnitude_product
    cosine_angle = np.clip(cosine_angle, -1.0, 1.0)

    angle = np.degrees(np.arccos(cosine_angle))

    return angle


def calculate_segment_angle(p1, p2):


    segment_vector = p2 - p1

    angle = np.arctan2(segment_vector[0], -segment_vector[1])

    return np.degrees(angle)


def calculate_side_view_features(main_side, landmark_data, const_person_height, torso_perc, thigh_perc, shin_perc):

    features = {}

    default_features = {key: None for key in [
        'mainHip_height_percent', 'mainKnee_height_percent', 'person_height',
        'torso_proportion_percent', 'thigh_proportion_percent', 'shin_proportion_percent',
        'angle_shin_thigh', 'angle_foot_shin', 'angle_core_thigh',
        'angle_thigh_horizontal', 'mainThigh_vertical_angle', 'mainCore_vertical_angle'
    ]}

    if main_side is None or not landmark_data or const_person_height is None or const_person_height <= 0:
        return default_features

    try:
        main_hip_coord = landmark_data[f'{main_side}_HIP'][:2]
        main_knee_coord = landmark_data[f'{main_side}_KNEE'][:2]
        main_ankle_coord = landmark_data[f'{main_side}_ANKLE'][:2]
        main_foot_index_coord = landmark_data[f'{main_side}_FOOT_INDEX'][:2]
        main_heel_coord = landmark_data[f'{main_side}_HEEL'][:2]
        main_shoulder_coord = landmark_data[f'{main_side}_SHOULDER'][:2]

        features['torso_proportion_percent'] = torso_perc
        features['thigh_proportion_percent'] = thigh_perc
        features['shin_proportion_percent'] = shin_perc
        features['person_height'] = const_person_height

        lowest_foot_y = max(main_heel_coord[1], main_foot_index_coord[1])

        features['mainKnee_angle'] = calculate_angle_3pt(main_hip_coord, main_knee_coord, main_ankle_coord)
        features['mainHip_angle'] = calculate_angle_3pt(main_shoulder_coord, main_hip_coord, main_knee_coord)
        features['mainAnkle_angle'] = calculate_angle_3pt(main_knee_coord, main_ankle_coord, main_foot_index_coord)

        thigh_vertical_angle = calculate_segment_angle(main_knee_coord, main_hip_coord)
        features['mainThigh_vertical_angle'] = thigh_vertical_angle
        core_vertical_angle = calculate_segment_angle(main_hip_coord, main_shoulder_coord)
        features['mainCore_vertical_angle'] = core_vertical_angle

        features['angle_thigh_horizontal'] = abs(90 - abs(thigh_vertical_angle))

        main_hip_height = lowest_foot_y - main_hip_coord[1]
        main_knee_height = lowest_foot_y - main_knee_coord[1]

        features['mainHip_height_percent'] = (main_hip_height / const_person_height) * 100
        features['mainKnee_height_percent'] = (main_knee_height / const_person_height) * 100

        features['angle_shin_thigh'] = features['mainKnee_angle']
        features['angle_foot_shin'] = features['mainAnkle_angle']
        features['angle_core_thigh'] = features['mainHip_angle']

        return features

    except Exception:
        return default_features


def calculate_time_from_min_to_max_hip_height(all_squat_features, frame_times):

    hip_heights_perc = [f.get('mainHip_height_percent') for f in all_squat_features]

    valid_data = [(i, h) for i, h in enumerate(hip_heights_perc) if h is not None]

    if len(valid_data) < 2:
        return None

    valid_indices = [item[0] for item in valid_data]
    valid_heights = np.array([item[1] for item in valid_data])

    min_hip_height = np.min(valid_heights)
    min_index_in_valid = np.where(valid_heights == min_hip_height)[0][0]
    min_frame_index = valid_indices[min_index_in_valid]
    min_time = frame_times[min_frame_index]

    heights_after_min = valid_heights[min_index_in_valid:]
    indices_after_min = valid_indices[min_index_in_valid:]

    if len(heights_after_min) < 2:
        return None

    max_frame_index = None

    for i in range(1, len(heights_after_min) - 1):
        prev_h = heights_after_min[i - 1]
        curr_h = heights_after_min[i]
        next_h = heights_after_min[i + 1]

        if (curr_h > prev_h and curr_h > next_h) or (curr_h >= prev_h and curr_h > next_h):
            max_frame_index = indices_after_min[i]
            break

    if max_frame_index is None:
        if len(indices_after_min) > 1:
            max_frame_index = indices_after_min[-1]
        else:
            return None

    max_time = frame_times[max_frame_index]

    concentration_time = max_time - min_time

    if concentration_time > 0.1 and max_time > min_time:
        return concentration_time
    else:
        return None


def export_side_view_analysis_csv(squat_features, median_torso, median_thigh, median_shin, torso_perc, thigh_perc,
                                  shin_perc, frame_times, out_path):
    with open(out_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)

        headers = [
            'frame_time_seconds',
            'mainHip_height_percent',
            'mainKnee_height_percent',
            'torso_proportion_percent',
            'thigh_proportion_percent',
            'shin_proportion_percent',
            'angle_shin_thigh',
            'angle_foot_shin',
            'angle_core_thigh',
            'angle_thigh_horizontal',
            'mainCore_vertical_angle'
        ]
        writer.writerow(headers)

        for frame_idx, frame_features in enumerate(squat_features):
            row = []

            row.append(frame_times[frame_idx] if frame_idx < len(frame_times) else None)

            row.append(frame_features.get('mainHip_height_percent'))
            row.append(frame_features.get('mainKnee_height_percent'))

            row.append(torso_perc)
            row.append(thigh_perc)
            row.append(shin_perc)

            row.append(frame_features.get('angle_shin_thigh'))
            row.append(frame_features.get('angle_foot_shin'))
            row.append(frame_features.get('angle_core_thigh'))
            row.append(frame_features.get('angle_thigh_horizontal'))
            row.append(frame_features.get('mainCore_vertical_angle'))

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
        return [], [], 0, 0

    fps = cap.get(cv2.CAP_PROP_FPS)
    initial_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    initial_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    rotation = 0

    try:
        result = subprocess.run(['ffprobe', '-i', video_path, '-show_streams', '-v', 'quiet', '-print_format', 'json'],
                                capture_output=True, text=True, check=False)
        probe_data = json.loads(result.stdout)
        if 'streams' in probe_data and len(probe_data['streams']) > 0:
            for stream in probe_data['streams']:
                if stream.get('codec_type') == 'video':
                    rotation = float(stream.get('side_data_list', [{}])[0].get('rotation', 0)) if stream.get(
                        'side_data_list') else 0
                    if rotation == 0:
                        rotation = float(stream.get('rotation', 0))
                    break
    except Exception:
        pass

    if abs(rotation) % 180 == 90:
        out_width, out_height = initial_height, initial_width
    else:
        out_width, out_height = initial_width, initial_height

    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    out = cv2.VideoWriter(output_path, fourcc, fps, (out_width, out_height))

    if not out.isOpened():

        cap.release()
        return [], [], 0, 0

    all_landmarks_normalized = []
    frame_times = []
    frame_count = 0
    VISIBILITY_THRESHOLD_DRAW = 0.3

    current_width, current_height = out_width, out_height

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1

        if rotation == 90:
            frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
        elif rotation == 180 or rotation == -180:
            frame = cv2.rotate(frame, cv2.ROTATE_180)
        elif rotation == 270 or rotation == -90:
            frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)

        frame = cv2.rotate(frame, cv2.ROTATE_180)

        current_height, current_width, _ = frame.shape

        current_time_msec = cap.get(cv2.CAP_PROP_POS_MSEC)
        current_time_sec = current_time_msec / 1000.0 if current_time_msec > 0 else frame_count / fps
        frame_times.append(current_time_sec)

        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(image_rgb)

        if results.pose_landmarks:
            pixel_landmarks_int = {}
            frame_landmarks_float = []

            for i, lm in enumerate(results.pose_landmarks.landmark):
                x_px_int = int(lm.x * current_width)
                y_px_int = int(lm.y * current_height)

                pixel_landmarks_int[i] = (x_px_int, y_px_int, lm.visibility)

                frame_landmarks_float.append([float(lm.x), float(lm.y), float(lm.z), float(lm.visibility)])

            all_landmarks_normalized.append(frame_landmarks_float)

            for connection in POSE_CONNECTIONS:
                start_idx = connection[0]
                end_idx = connection[1]

                pt1_data = pixel_landmarks_int.get(start_idx)
                pt2_data = pixel_landmarks_int.get(end_idx)

                if pt1_data and pt2_data:
                    pt1 = pt1_data[:2]
                    pt2 = pt2_data[:2]
                    vis1 = pt1_data[2]
                    vis2 = pt2_data[2]

                    if vis1 > VISIBILITY_THRESHOLD_DRAW and vis2 > VISIBILITY_THRESHOLD_DRAW:
                        cv2.line(frame, pt1, pt2, (0, 255, 0), 2)

            for x, y, vis in pixel_landmarks_int.values():
                if vis > VISIBILITY_THRESHOLD_DRAW:
                    cv2.circle(frame, (x, y), 5, (0, 255, 255), -1)

        else:
            all_landmarks_normalized.append(None)

        out.write(frame)

    cap.release()
    out.release()

    return all_landmarks_normalized, frame_times, current_width, current_height


def process_videos():
    pose = mp_pose.Pose(static_image_mode=False, model_complexity=2, enable_segmentation=False,
                        min_detection_confidence=0.5, min_tracking_confidence=0.5)

    video_dir = "../squat_clips_acc"
    output_dir = "../stick_figures_acc"

    os.makedirs(output_dir, exist_ok=True)

    if not os.path.isdir(video_dir) or not os.listdir(video_dir):
        return

    for file_name in os.listdir(video_dir):
        video_extensions = ('.mp4', '.mov', '.avi', '.mkv', '.MP4', '.MOV', '.AVI', '.MKV')
        if not file_name.lower().endswith(tuple(ext.lower() for ext in video_extensions)):
            continue

        video_path = os.path.join(video_dir, file_name)

        base_name = os.path.splitext(file_name)[0]
        temp_output_path = os.path.join(output_dir, f"{base_name}_acc_stick_temp.avi")
        trajectory_path = os.path.join(output_dir, f"{base_name}_acc_analysis.csv")

        all_landmarks_normalized, frame_times, width, height = processing(video_path, temp_output_path, pose)

        if not all_landmarks_normalized or width == 0:
            if os.path.exists(temp_output_path):
                os.remove(temp_output_path)
            continue

        interpolated_landmarks = interpolate_missing_landmarks(all_landmarks_normalized)
        smoothed_landmarks = apply_temporal_smoothing(interpolated_landmarks, window_size=5)

        height_samples = []
        torso_lengths = []
        thigh_lengths = []
        shin_lengths = []

        for frame_landmarks_normalized in smoothed_landmarks:
            if frame_landmarks_normalized is not None:
                main_side, current_landmark_data = calculate_body_side_view(
                    frame_landmarks_normalized, width, height, visibility_threshold=0.0
                )

                if current_landmark_data:
                    height_tuple = calculate_segment_lengths_and_height_factor(current_landmark_data)

                    if height_tuple is not None:
                        height_samples.append(height_tuple[0])
                        torso_lengths.append(height_tuple[1])
                        thigh_lengths.append(height_tuple[2])
                        shin_lengths.append(height_tuple[3])

        if height_samples:
            CONST_PERSON_HEIGHT = np.median(height_samples)
            MEDIAN_TORSO_LENGTH = np.median(torso_lengths)
            MEDIAN_THIGH_LENGTH = np.median(thigh_lengths)
            MEDIAN_SHIN_LENGTH = np.median(shin_lengths)

            TORSO_PROP_PERC = (MEDIAN_TORSO_LENGTH / CONST_PERSON_HEIGHT) * 100
            THIGH_PROP_PERC = (MEDIAN_THIGH_LENGTH / CONST_PERSON_HEIGHT) * 100
            SHIN_PROP_PERC = (MEDIAN_SHIN_LENGTH / CONST_PERSON_HEIGHT) * 100
        else:
            CONST_PERSON_HEIGHT = height if height > 0 else 480
            MEDIAN_TORSO_LENGTH = CONST_PERSON_HEIGHT * 0.30
            MEDIAN_THIGH_LENGTH = CONST_PERSON_HEIGHT * 0.25
            MEDIAN_SHIN_LENGTH = CONST_PERSON_HEIGHT * 0.25
            TORSO_PROP_PERC = 30.0
            THIGH_PROP_PERC = 25.0
            SHIN_PROP_PERC = 25.0

        all_squat_features = []

        ANALYSIS_VISIBILITY_THRESHOLD = 0.0

        for frame_landmarks_normalized in smoothed_landmarks:
            main_side, current_landmark_data = calculate_body_side_view(
                frame_landmarks_normalized, width, height, visibility_threshold=ANALYSIS_VISIBILITY_THRESHOLD
            )

            if current_landmark_data:
                height_features = calculate_side_view_features(
                    main_side, current_landmark_data, CONST_PERSON_HEIGHT,
                    TORSO_PROP_PERC, THIGH_PROP_PERC, SHIN_PROP_PERC
                )
                all_squat_features.append(height_features)
            else:
                all_squat_features.append({key: None for key in [
                    'mainHip_height_percent', 'mainKnee_height_percent', 'person_height',
                    'torso_proportion_percent', 'thigh_proportion_percent', 'shin_proportion_percent',
                    'angle_shin_thigh', 'angle_foot_shin', 'angle_core_thigh',
                    'angle_thigh_horizontal', 'mainThigh_vertical_angle', 'mainCore_vertical_angle'
                ]})

        export_side_view_analysis_csv(
            all_squat_features, MEDIAN_TORSO_LENGTH, MEDIAN_THIGH_LENGTH, MEDIAN_SHIN_LENGTH,
            TORSO_PROP_PERC, THIGH_PROP_PERC, SHIN_PROP_PERC, frame_times, trajectory_path
        )

        time_min_to_max = calculate_time_from_min_to_max_hip_height(all_squat_features, frame_times)

        final_output_path = temp_output_path

        if time_min_to_max is not None:
            time_str = f"T_{time_min_to_max:.3f}s"
            new_file_name = f"{base_name}_{time_str}_acc_stick.avi"
            final_output_path = os.path.join(output_dir, new_file_name)

            try:
                os.rename(temp_output_path, final_output_path)
            except OSError:
                final_output_path = temp_output_path
        else:
            final_output_path = os.path.join(output_dir, f"{base_name}_acc_stick.avi")
            if os.path.exists(temp_output_path):
                os.rename(temp_output_path, final_output_path)


if __name__ == "__main__":
    process_videos()