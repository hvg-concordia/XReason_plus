import numpy as np
import re
import joblib

def euclidean_distance(sample1, sample2):

    # Convert samples to numpy arrays for element-wise operations
    sample1 = np.array(sample1)
    sample2 = np.array(sample2)

    # Calculate the Euclidean distance
    distance = np.sqrt(np.sum((sample1 - sample2) ** 2))
    return distance

def prediction_adv(model_path,perturbed_sample):
    model = joblib.load(model_path)
    # Make predictions for all instances in the test data
    predictions = model.predict(perturbed_sample)[0]
    return predictions

def Determine_nb_features(path):
    all_features=[]
    Targets=[]
    f = path
    result={}
    index=-1
    with open(f, 'r') as file:
        for line_number, line in enumerate(file, start=1):
             if 'explanation' in line:
                # print(line.strip())
                explanation=line.strip()
                match = re.search(r'IF (.*?) THEN (\d+)', explanation)
                if match:
                    condition_str = match.group(1)
                    class_label = int(match.group(2))
                    Targets.append(class_label)
                    # Split conditions and extract variable-value pairs
                    # conditions = []
                    d={}
                    for condition in condition_str.split(' AND '):
                        var, val = map(str.strip, condition.split('=='))
                        d[var]=float(val)
                        # conditions.append(d)
                    index+=1
                    if class_label in result:
                        result[class_label].append([index,d])
                    else:
                        result[class_label] = [[index,d]]

    return result,Targets

def extract_features_from_line(line, features_to_extract):
    pattern = re.compile(r"(Feat\d+)\s*==\s*([-+]?\d*\.?\d+)")
    matches = pattern.findall(line)
    extracted = {}
    for match in matches:
        feature, value = match
        if feature in features_to_extract:
            extracted[feature] = float(value)
    return extracted

def extract_features_from_file(file_path, features_dict, features_to_extract):
    results = []

    with open(file_path, 'r') as file:
        while True:
            data_line = file.readline().strip()
            if not data_line:
                break
            explanation_line = file.readline().strip()
            extracted_features = extract_features_from_line(explanation_line, features_to_extract)
            results.append(extracted_features)

    # Filter the results based on features_dict
    filtered_results = []
    for extracted_features in results:
        filtered_result = {feature: features_dict[feature] for feature in extracted_features if feature in features_dict}
        filtered_results.append(filtered_result)

    return filtered_results

import random

def get_outside_value(value, intervals,column_interval):
    """
    Generate a new value outside the given intervals
    :param value: The original value.
    :param intervals: List of intervals to avoid.
    :return: A new value outside the intervals.
    """
    def is_in_intervals(val, intervals):
        for interval in intervals:
            if isinstance(interval, list):
                if len(interval) == 1:
                    if val == interval[0]:
                        return True
                elif len(interval) == 2:
                    if interval[0] <= val <= interval[1]:
                        return True
        return False

    range_width = column_interval[1] - column_interval[0]
    base_fraction = 0.080009  # Default 5%
    scaling_factor = 0.45
    perturbation_factor = base_fraction * (range_width / (range_width + scaling_factor))
    # Ensure the fraction is not too small or too large
    perturbation_factor = min(max(perturbation_factor, 0.080009), 0.080009)
    for _ in range(100000):
        new_value = value + random.uniform(-perturbation_factor * range_width, perturbation_factor * range_width)
        if not is_in_intervals(new_value, intervals):
            return new_value
        # Gradually increase the perturbation
        perturbation_factor += 0.080009
    # If no value found after many attempts, raise an error
    raise ValueError("Unable to find a suitable outside value")

def perturb_sample(sample, important_intervals,data,features_with_indexes):
    """
    Perturb the important features of the sample to be outside the given intervals.

    :param sample: List of feature values.
    :param important_intervals: Dictionary with feature names as keys and intervals as values.
    :return: Perturbed sample.
    """
    perturbed_sample = sample.copy().reshape(-1, 1)
    sample=sample.reshape(-1, 1)
    for feature, intervals in important_intervals.items():
        for feat, index in features_with_indexes.items():
          if feat == feature:
            feature_index = index
            break
        print("feature_index",feature_index)
        min_interval=min(data.iloc[:, feature_index].values)
        print("min_interval",min_interval)
        max_interval=max(data.iloc[:, feature_index].values)
        column_interval=[min_interval,max_interval]
        print("column_interval",column_interval)
        perturbed_sample[feature_index] = get_outside_value(sample[feature_index], intervals,column_interval)
    return perturbed_sample

def is_within_interval(value, interval):
    if len(interval) == 1:
        return value == interval[0]
    elif len(interval) == 2:
        lower_bound, upper_bound = interval
        return lower_bound <= value <= upper_bound
    
def extract_feature_indices(lines, feature_index_map):
    feature_indices_list = []
    for line in lines:
        indices = []
        for feature in feature_index_map:
            if feature in line:
                indices.append(feature_index_map[feature])
        feature_indices_list.append(indices)
    return feature_indices_list