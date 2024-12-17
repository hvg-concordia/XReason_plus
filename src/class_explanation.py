import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.exceptions import NotFittedError
import re
from collections import defaultdict

def find_optimal_clusters(data, max_k=10):
    data = np.array(data).reshape(-1, 1)
    n_samples = len(data)

    if n_samples == 1:
        return 1

    if n_samples == 2:
        return 2

    max_k = min(max_k, n_samples)
    best_n_clusters = 2
    best_score = -np.inf

    for n_clusters in range(2, max_k + 1):
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        try:
            labels = kmeans.fit_predict(data)
            if len(np.unique(labels)) < 2:
                continue
            score = silhouette_score(data, labels)
            if score > best_score:
                best_score = score
                best_n_clusters = n_clusters
        except (NotFittedError, ValueError):
            continue

    return best_n_clusters if best_score > -np.inf else 2

def cluster_extraction(feature_values):
    feature_values = np.array(feature_values).reshape(-1, 1)
    optimal_n_clusters = find_optimal_clusters(feature_values)
    # print(f"Optimal number of clusters: {optimal_n_clusters}")

    if len(feature_values) == 1:
        return [[feature_values[0][0], feature_values[0][0]]]

    kmeans = KMeans(n_clusters=optimal_n_clusters, random_state=42, n_init=10)
    kmeans.fit(feature_values)

    cluster_centers = kmeans.cluster_centers_.flatten()
    labels = kmeans.labels_

    sorted_indices = np.argsort(cluster_centers)
    sorted_labels = np.array([sorted_indices[label] for label in labels])

    intervals = []
    for cluster_id in range(optimal_n_clusters):
        cluster_values = feature_values[sorted_labels == cluster_id].flatten()
        if len(cluster_values) > 0:
            intervals.append([np.min(cluster_values), np.max(cluster_values)])

    # print(intervals)
    return intervals

# create dictionary {feature: [extracted values for specific class]}
def todict(maxsat_list):
    result_dict = {}

    for sub_list in maxsat_list:
        for pair in sub_list:
            key, value = pair
            if key in result_dict:
                result_dict[key].append(float(value))
            else:
                result_dict[key] = [float(value)]
    return result_dict



def interval_extraction(column_data):
    # Calculate the differences between consecutive points
    differences = [column_data[i+1] - column_data[i] for i in range(len(column_data)-1)]
    # Calculate the average difference
    average_difference = sum(differences) / len(set(differences)) if differences else 0
    # Create intervals based on the average difference
    intervals = []
    current_interval = [column_data[0]]

    for i in range(len(differences)):
        if differences[i] > average_difference:
            # Start a new interval
            intervals.append(current_interval)
            current_interval = [column_data[i+1]]
        else:
            # Extend the current interval
            if len(current_interval) >= 2:
            # Replace the second element in the last interval
                current_interval[-1] = column_data[i+1]
            else:
                current_interval.append(column_data[i+1])

    # Add the last interval
    intervals.append(current_interval)
    # print("recover",intervals)
    return intervals

def parse_data(file_content):
    lines = file_content.strip().split('\n')
    # data = []
    explanations = []
    for i in range(0, len(lines)):
        # features = list(map(float, lines[i].split(',')))
        explanation = lines[i]
        # data.append(features)
        explanations.append(explanation)
    return  explanations

def extract_conditions(explanation):
    # Split the explanation into conditions and result
    conditions_str, result = explanation.split(" THEN ")

    # Remove the leading "IF" and split into individual conditions
    conditions = conditions_str.replace("IF ", "").split(" AND ")
    # Process each condition
    extracted = []
    for condition in conditions:
    # Remove 'explanation: "' if present
      cleaned_condition = condition.replace('explanation: "', '').strip()

      # Match both feature name and value
      match = re.match(r'([\w\s]+) == ([-+]?\d*\.\d+|\d+)', cleaned_condition)

      if match:
          feature, value = match.groups()
          extracted.append([feature.strip(), float(value)])
    return extracted

def extract_target(explanation):
    match = re.search(r'THEN (\d+)', explanation)
    return int(match.group(1)) if match else None

def organize_by_class(explanations):
    class_dict = defaultdict(list)
    for explanation in explanations:
        target_class = extract_target(explanation)
        if target_class is not None:
            conditions = extract_conditions(explanation)
            class_dict[target_class].append(conditions)
    return class_dict

def main(explanations_path):
    # Read the data from the text file
    with open(explanations_path, 'r') as file:
        file_content = file.read()

    explanations = parse_data(file_content)
    print(explanations)
    class_dict = organize_by_class(explanations)
    # Print the results
    # for key, value in class_dict.items():
    #     print(f'{key}: {value}')
    return  class_dict

def generate_class_exp(explanations_path):
  class_dict=main(explanations_path)
  classes=list(class_dict.keys()) # Convert dict_keys to a list
  i=0
  results_generate={}
  results_recover={}
  for classe in class_dict.values():
    result_dict=todict(classe)
    result_dict_recover=todict(classe)
    for features,lists in result_dict.items():
        lists=list(set(lists))
        lists.sort()
        l_generate=cluster_extraction(lists)
        l_recover=interval_extraction(lists)
        result_dict[features]=l_generate
        result_dict_recover[features]=l_recover
        results_generate[classes[i]]=result_dict
        results_recover[classes[i]]=result_dict_recover

        # yield("class",classes[i],":",result_dict)
    i+=1
  return results_generate,results_recover