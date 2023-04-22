from calendar import c
import math
from turtle import distance
from pyspark import SparkConf, SparkContext
import sys
import time
import numpy as np
from collections import defaultdict
from sklearn.cluster import KMeans

def write_csv_file(result_str, output_file):
    with open(output_file, "w") as f:
        result_header = "user_id, business_id, prediction\n"
        f.writelines(result_header)
        f.writelines(result_str)

def mahanabolis_dis(point, cluster):
    N, SUM, SUMSQ = cluster
    centroid = SUM / N
    sigma = (SUMSQ / N) - (SUM / N) ** 2
    z = (point[1] - centroid)/sigma
    m_distance = np.dot(z, z) ** (1/2)
    return m_distance

if __name__ == "__main__":
    sc = SparkContext()

    # input_file = sys.argv[1]
    # n_clusters = int(sys.argv[2])
    # output_file = sys.argv[3]

    input_file = "BFR-algorithm/data/hw6_clustering.txt"
    n_clusters = 15
    output_file = "BFR-algorithm/result/task.csv"

    d = 0

    data = sc.textFile(input_file)

    data = data.map(lambda x: x.strip("\n").split(',')) \
                    .map(lambda x: (int(x[0]), np.array(x[2:], dtype=np.float64)))
    
    features_array = data.first()[1]
    d = len(features_array)

    # Step 1. Load 20% of the data randomly.
    data = data.randomSplit([0.2, 0.8], seed=42)

    chunk_data = data[0]
    chunk_len = len(chunk_data.collect())
    remaining_data = data[1]

    # print(len(chunk_data.collect()))
    # print(len(remaining_data.collect()))

    # Step 2. Run K-Means with a large K 
    K = 5 * n_clusters

    vector_list = chunk_data.map(lambda x: x[1]).collect()

    k_means = KMeans(n_clusters = K, n_init = 10).fit(vector_list)

    # Step 3. Move all the clusters that contain only one point to RS

    labels = k_means.labels_
    counts = np.bincount(k_means.labels_)

    cluster_dict = defaultdict(list)

    for idx, c_id in enumerate(labels):
        cluster_dict[c_id].append(idx)
    
    rs_indices = [idx[0] for idx in cluster_dict.values() if len(idx) == 1]

    chunk_with_RS = [vector_list[i] for i in rs_indices]

    chunk_without_RS = np.delete(np.array(vector_list), rs_indices, axis=0)

    # Step 4. Run K-Means again to cluster the rest of the data points with K = the number of input clusters
    
    K = n_clusters
    k_means = KMeans(n_clusters = K, n_init = 10).fit(chunk_without_RS)


    # Step 5. Use the K-Means result from Step 4 to generate the DS clusters

    labels = k_means.labels_
    clusters = defaultdict(list)
    ds_clusters = defaultdict(list)
    ds_point_dict = defaultdict(list)
    ds_centroid_dict = defaultdict(list)
    ds_distance_dict = defaultdict(list)

    for idx, c_id in enumerate(labels):
        clusters[c_id].append(idx)

    for c_id, idx in clusters.items():
        N = len(idx)

        feature_vectors = chunk_without_RS[idx, :]
        SUM = np.sum(feature_vectors, axis=0)
        SUMSQ = np.sum(np.square(feature_vectors), axis=0)

        points = np.array(chunk_without_RS[idx, 0]).astype(int).tolist()
        centroid = SUM / N

        distance_from_centroid = np.sqrt(np.subtract(SUMSQ / N, np.square(centroid)))

        ds_clusters[c_id] = [N, SUM, SUMSQ]
        ds_point_dict[c_id] = points
        ds_centroid_dict[c_id] = centroid
        ds_distance_dict[c_id] = distance_from_centroid

    # print(len(ds_clusters))

#     Step 6. Run K-Means on the points in the RS with a large K (e.g., 5 times of the number of the input
# clusters) to generate CS (clusters with more than one points) and RS (clusters with only one point).

    cs_clusters = defaultdict(list)

    if len(rs_indices) >= 5 * n_clusters:
        K_rs = 5 * n_clusters
        k_means_rs = KMeans(n_clusters = K_rs, n_init = 10).fit(chunk_with_RS)

        labels = k_means_rs.labels_
        cluster_dict = defaultdict(list)

        for idx, c_id in enumerate(labels):
            cluster_dict[c_id].append(idx)

        rs_indices = [idx[0] for idx in cluster_dict.values() if len(idx) == 1]


        cs_clusters = defaultdict(list)
        cs_point_dict = defaultdict(list)
        cs_centroid_dict = defaultdict(list)
        cs_distance_dict = defaultdict(list)

        for c_id, idx in cluster_dict.items():
            N = len(idx)

            feature_vectors = chunk_with_RS[idx, :]
            SUM = np.sum(feature_vectors, axis=0)
            SUMSQ = np.sum(np.square(feature_vectors), axis=0)
            points = np.array(chunk_without_RS[idx, 0]).astype(int).tolist()
            centroid = SUM / N
            distance_from_centroid = np.sqrt(np.subtract(SUMSQ / N, np.square(centroid)))

            cs_clusters[c_id] = [N, SUM, SUMSQ]
            cs_point_dict[c_id] = points
            cs_centroid_dict[c_id] = centroid
            cs_distance_dict[c_id] = distance_from_centroid
        
    
    num_d = 0
    num_c = 0

    for value in ds_clusters.values():
        num_d += value[0]

    for value in cs_clusters.values():
        num_c += value[0]

    result_str = "The intermediate results:\n"
    result_str += 'Round 1: ' + str(num_d) + ',' + str(len(cs_clusters)) + ',' + str(num_c) + ',' + str(len(rs_indices)) + '\n'



    #Step 7-12

    remaining_data = remaining_data.collect()
    for i in range(4):

        #Step 7: Load another 20% of the data randomly
        chunk = remaining_data[:chunk_len]
        remaining_data = remaining_data[chunk_len:]

        # Step 8. For the new points, compare them to each of the DS using the Mahalanobis Distance and assign them to the nearest DS clusters if the distance is < 2 root(𝑑).

        D = 2 * math.sqrt(d)
        cluster = -1

        for i in range(chunk_len):
            point = chunk[i]
            distance_dict = dict()

            for c_id, values in ds_clusters.items():
                distance_dict[c_id] = mahanabolis_dis(point, values)  

            max_distance = min(list(distance_dict.values()))

            for c_id in distance_dict:
                if distance_dict[c_id] == max_distance:
                    cluster = c_id
        
        print(type(chunk))
        if max_distance < D and cluster != -1:
            N, SUM, SUMSQ = ds_clusters[cluster]
            N += 1
            SUM = np.add(SUM, chunk[1])
            SUMSQ = np.add(SUMSQ, np.square(chunk[1]))

            centroid = SUM / N
            distance_from_centroid = np.sqrt(np.subtract(SUMSQ / N, np.square(centroid)))

            ds_clusters[cluster] = [N, SUM, SUMSQ]
            ds_centroid_dict[cluster] = centroid
            ds_distance_dict[cluster] = distance_from_centroid
            ds_point_dict[cluster].append(int(chunk[0]))

            print(ds_point_dict)

        

                
