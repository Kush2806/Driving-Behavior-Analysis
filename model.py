import pandas as pd
import numpy as np
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean

def calculate_driving_score(event_counts):
    baseline_score = 100
    def get_deduction(count, low, medium, high):
        if count <= 2:
            return low
        elif 3 <= count <= 5:
            return medium
        else:
            return high

    harsh_braking = get_deduction(event_counts.get('Harsh Braking', 0), -2, -5, -10)
    harsh_acceleration = get_deduction(event_counts.get('Harsh Acceleration', 0), -2, -5, -10)
    harsh_left_turn = get_deduction(event_counts.get('Harsh Left', 0), -1, -3, -7)
    harsh_right_turn = get_deduction(event_counts.get('Harsh Right', 0), -1, -3, -7)

    total_deductions = harsh_braking + harsh_acceleration + harsh_left_turn + harsh_right_turn
    return max(baseline_score + total_deductions, 0)

def process_driving_data(trip_data, left_template_file, right_template_file):
    if isinstance(trip_data, pd.DataFrame):
        pass
    else:  # Otherwise, assume it's a file path and load the data
        trip_data = pd.read_csv(trip_data)
    normal_left_template = pd.read_csv(left_template_file)[['Gyroscope_X', 'Gyroscope_Y', 'Gyroscope_Z']].values
    normal_right_template = pd.read_csv(right_template_file)[['Gyroscope_X', 'Gyroscope_Y', 'Gyroscope_Z']].values

    maneuver_window_size = 800
    maneuver_shift_size = 50
    acc_window_size = 15
    acc_shift_size = 15
    gyro_distance_threshold = 50
    harsh_gyro_distance_threshold = 150
    harsh_acceleration_threshold = 2.5
    harsh_braking_threshold = -3.0
    harsh_left_acc_threshold = -2.0
    harsh_right_acc_threshold = 2.0

    maneuver_events = []
    acceleration_events = []
    event_counts = {
        'Normal Left': 0, 
        'Normal Right': 0, 
        'Harsh Left': 0, 
        'Harsh Right': 0, 
        'Harsh Acceleration': 0, 
        'Harsh Braking': 0
    }

    for start in range(0, len(trip_data) - maneuver_window_size + 1, maneuver_shift_size):
        end = start + maneuver_window_size
        window_data_gyro = trip_data[['Gyroscope_X', 'Gyroscope_Y', 'Gyroscope_Z']].iloc[start:end].values
        window_data_acc = trip_data[['Accelerometer_X', 'Accelerometer_Y', 'Accelerometer_Z']].iloc[start:end].values
        left_distance, _ = fastdtw(window_data_gyro, normal_left_template, dist=euclidean)
        right_distance, _ = fastdtw(window_data_gyro, normal_right_template, dist=euclidean)

        distances = {'Left': left_distance, 'Right': right_distance}
        min_maneuver = min(distances, key=distances.get)
        min_distance = distances[min_maneuver]
        event = 'No Turning'

        if min_distance <= gyro_distance_threshold:
            event = f'Normal {min_maneuver}'
        elif min_distance <= harsh_gyro_distance_threshold:
            avg_acc_x = np.mean(window_data_acc[:, 0])
            if avg_acc_x <= harsh_left_acc_threshold:
                event = 'Harsh Left'
            if avg_acc_x >= harsh_right_acc_threshold:
                event = 'Harsh Right'

        maneuver_events.append((start, event, min_distance))

    for start in range(0, len(trip_data) - acc_window_size + 1, acc_shift_size):
        end = start + acc_window_size
        window_acc_y = trip_data['Accelerometer_Y'].iloc[start:end]
        avg_acc_y = np.mean(window_acc_y)
        if avg_acc_y > harsh_acceleration_threshold:
            event = 'Harsh Acceleration'
        elif avg_acc_y < harsh_braking_threshold:
            event = 'Harsh Braking'
        else:
            event = 'No Event'

        acceleration_events.append((start, event, avg_acc_y))

    def remove_duplicates(events):
        filtered_events = []
        previous_event = None
        for index, event, value in events:
            if event != previous_event or event in ['No Turning', 'No Event']:
                filtered_events.append((index, event, value))
                if event in event_counts:
                    event_counts[event] += 1
            previous_event = event
        return filtered_events

    remove_duplicates(maneuver_events)
    remove_duplicates(acceleration_events)

    return {
        'event_counts': event_counts,
        'driving_score': calculate_driving_score(event_counts)
    }
