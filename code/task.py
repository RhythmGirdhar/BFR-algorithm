import sys
import numpy as np
from collections import defaultdict
from sklearn.cluster import KMeans
import itertools

def write_csv(output_file, result_str, result):
    with open(output_file, "w") as f:
        result_str += "\n"
        result_str += "The clustering results:\n"
        f.write(result_str)
        for point in sorted(result.keys()):
            f.write(str(point) + ',' + str(result[point]) + '\n')           
    return

def create_cluster_dict(labels):
    cluster_dict = defaultdict(list)   
    for idx, c_id in enumerate(labels):
        cluster_dict[c_id].append(idx)
    return cluster_dict

def calculate_stats(cs_cluster, ds_cluster):
    num_d = sum(value[0] for value in ds_cluster.values())
    num_c = sum(value[0] for value in cs_cluster.values())  
    return num_c, num_d

def get_cluster_statistics(data, clusters, flag):
    cluster = defaultdict(list)
    point_dict = defaultdict(list)
    centroid_dict = defaultdict(list)
    deviation_dict = defaultdict(list)

    if flag == "CS":
        for cid, idx in clusters.items():
            if len(idx) > 1:
                feature = data[idx, 2:]
                N = len(idx)
                SUM = np.sum(feature, axis = 0)
                SUMSQ = np.sum(np.square(feature), axis = 0)

                points = np.array(data[idx, 0]).astype(int).tolist()
                centroid = SUM / N
                deviation = np.sqrt((SUMSQ / N) - np.square(centroid))

                cluster[cid] = [N, SUM, SUMSQ]
                point_dict[cid] = points
                centroid_dict[cid] = centroid
                deviation_dict[cid] = deviation

    elif flag == "DS":
        for cid, idx in clusters.items():
            feature = data[idx, 2:]
            N = len(idx)
            SUM = np.sum(feature, axis = 0)
            SUMSQ = np.sum(np.square(feature), axis = 0)

            points = np.array(data[idx, 0]).astype(int).tolist()
            centroid = SUM / N
            deviation = np.sqrt((SUMSQ / N) - np.square(centroid))

            cluster[cid] = [N, SUM, SUMSQ]
            point_dict[cid] = points
            centroid_dict[cid] = centroid
            deviation_dict[cid] = deviation

    return cluster, point_dict, centroid_dict, deviation_dict

def calculate_min_distance(given_cluster, centroid_dict, deviation_dict):
    maxx = float('inf')
    cluster = -1
    for cid in given_cluster.keys():
        mahalanobis_dis = mahalanobis_point_cluster(data, centroid_dict[cid], deviation_dict[cid])
        if mahalanobis_dis < maxx:
            maxx = mahalanobis_dis
            cluster = cid
    
    return maxx, cluster

def update_cluster_statistics(chunk, cid, value, cluster, point_dict, centroid_dict, deviation_dict):
    N = cluster[cid][0] + 1
    SUM = np.add(cluster[cid][1], chunk)
    SUMSQ = np.add(cluster[cid][2], np.square(chunk))

    centroid = SUM / N
    deviation = np.sqrt((SUMSQ / N) - np.square(centroid))

    cluster[cid] = [N, SUM, SUMSQ]
    centroid_dict[cid] = centroid
    deviation_dict[cid] = deviation

    point_dict[cid].append(int(value[0]))

    return cluster, point_dict, centroid_dict, deviation_dict

def mahalanobis_point_cluster(data, centroid, deviation):
    mahalanobis_dis = np.sqrt(np.sum(np.square(np.divide(np.subtract(data, centroid), deviation)), axis=0))
    return mahalanobis_dis

def mahalanobis_cluster_cluster(centroid1, centroid2, deviation1, deviation2):

    mahalanobis_dis1 = np.sqrt(np.sum(np.square(np.divide(np.subtract(centroid1, centroid2), deviation2, out=np.zeros_like(np.subtract(centroid1, centroid2)), where=deviation2 != 0)), axis=0))
    mahalanobis_dis2 = np.sqrt(np.sum(np.square(np.divide(np.subtract(centroid2, centroid1), deviation1, out=np.zeros_like(np.subtract(centroid2, cs_centroid[cid1])), where=deviation1 != 0)), axis=0))
    mahalanobis_dis = min(mahalanobis_dis1, mahalanobis_dis2)

    return mahalanobis_dis

def merged_cluster_statistics(cluster_1, cluster_2):

    N = cluster_1[0] + cluster_2[0]
    SUM = np.add(cluster_1[1], cluster_2[1])
    SUMSQ = np.add(cluster_1[2], cluster_2[2])

    centroid = SUM / N
    deviation = np.sqrt(np.subtract(SUMSQ / N, np.square(centroid)))

    return [N,SUM,SUMSQ], centroid, deviation
   
if __name__ == "__main__":

    input_file = sys.argv[1]
    n_clusters = int(sys.argv[2])
    output_file = sys.argv[3]

    # input_file = "BFR-algorithm/data/hw6_clustering.txt"
    # n_clusters = 10
    # output_file = "BFR-algorithm/result/task.csv"

    #variables
    NO_OF_CHUNKS = 5

    #initializations
    RS = set()

    ds_cluster = defaultdict(list)
    ds_point = defaultdict(list)
    ds_centroid = defaultdict(list)  
    ds_deviation = defaultdict(list)

    cs_cluster = defaultdict(list)
    cs_point = defaultdict(list)
    cs_centroid = defaultdict(list)  
    cs_deviation = defaultdict(list)

    D = 0

    np.random.seed(42)

    npdata = np.genfromtxt(input_file, delimiter=',')

    # Step 1. Load 20% of the data randomly.

    np.random.shuffle(npdata)
    npdata = np.array_split(npdata, NO_OF_CHUNKS)
    first_chunk = npdata[0]

    # Step 2. Run K-Means with a large K 

    K = 5 * n_clusters
    features = first_chunk[:, 2:]
    k_means_1 = KMeans(n_clusters = K).fit(features)

    # Step 3. Move all the clusters that contain only one point to RS

    cluster_dict = create_cluster_dict(k_means_1.labels_)
    rs = set([idx[0] for idx in cluster_dict.values() if len(idx) == 1])

    without_rs = np.delete(first_chunk, list(rs), axis = 0)

    # Step 4. Run K-Means again to cluster the rest of the data points with K = the number of input clusters
    
    K = n_clusters
    features = without_rs[:, 2:]
    k_means_2 = KMeans(n_clusters = K).fit(features)

    # Step 5. Use the K-Means result from Step 4 to generate the DS clusters

    cluster_dict = create_cluster_dict(k_means_2.labels_)
    ds_cluster, ds_point, ds_centroid, ds_deviation = get_cluster_statistics(without_rs, cluster_dict, "DS")
    
    #     Step 6. Run K-Means on the points in the RS with a large K (e.g., 5 times of the number of the input
# clusters) to generate CS (clusters with more than one points) and RS (clusters with only one point).

    with_rs = first_chunk[list(rs), :]
    K = 5 * n_clusters

    if len(rs) >= K :
        features = with_rs[:, 2:]
        k_means_3 = KMeans(n_clusters = K).fit(features)

        cluster_dict = create_cluster_dict(k_means_3.labels_)
        rs = set([idx[0] for idx in cluster_dict.values() if len(idx) == 1])

        cs_cluster, cs_point, cs_centroid, cs_deviation = get_cluster_statistics(with_rs, cluster_dict, "CS")
    
    num_c, num_d = calculate_stats(cs_cluster, ds_cluster)

    result_str = "The intermediate results:\n"
    result_str += 'Round 1: ' + str(num_d) + ',' + str(len(cs_cluster)) + ',' + str(num_c) + ',' + str(len(rs)) + '\n'


    #Step 7 - 12
    D = 2 * np.sqrt(first_chunk.shape[1] - 2)

    for i in range(1, 5):
        #Step 7
        chunk = npdata[i]
        for idx, value in enumerate(chunk):
            data = value[2:]

            #Step 8
            maxx, cluster = calculate_min_distance(ds_cluster, ds_centroid, ds_deviation)  

            if maxx < D and cluster != -1:
                ds_cluster, ds_point, ds_centroid, ds_deviation = update_cluster_statistics(data, cluster, value, ds_cluster, ds_point, ds_centroid, ds_deviation)
                
            else:

                #Step 9
                maxx, cluster = calculate_min_distance(cs_cluster, cs_centroid, cs_deviation)

                if maxx < D and cluster != -1:
                    cs_cluster, cs_point, cs_centroid, cs_deviation = update_cluster_statistics(data, cluster, value, cs_cluster, cs_point, cs_centroid, cs_deviation)

                else:
                    #step 10
                    rs.add(idx)

        #step 11

        data_rs = chunk[list(rs), :]
        K = 5 * n_clusters

        if len(rs) >= K:

            features = data_rs[:, 2:]
            k_means_4 = KMeans(n_clusters = K).fit(features)

            cluster_dict = create_cluster_dict(k_means_4.labels_)
            rs = set([idx[0] for idx in cluster_dict.values() if len(idx) == 1])
                    
            cs_cluster, cs_point, cs_centroid, cs_deviation = get_cluster_statistics(data_rs, cluster_dict, "CS")

        #step 12
        merged_cs_cs_clusters = dict()
        cluster = -1
        for cid1, cid2 in itertools.combinations(cs_cluster.keys(), 2):
            mahalanobis_dis = mahalanobis_cluster_cluster(cs_centroid[cid1], cs_centroid[cid2], cs_deviation[cid1], cs_deviation[cid2])
            if mahalanobis_dis < D:
                D = mahalanobis_dis
                cluster = cid2

            merged_cs_cs_clusters[cid1] = cluster
 
        for cid1, cid2 in merged_cs_cs_clusters.items():
            if cid1 in cs_cluster and cid2 in cs_cluster:
                if cid1 != cid2:

                    stats, centroid, deviation = merged_cluster_statistics(cs_cluster[cid1], cs_cluster[cid2])

                    cs_cluster[cid2] = stats
                    cs_centroid[cid2] = centroid
                    cs_deviation[cid2] = deviation
                    cs_point[cid2].extend(cs_point[cid1])
           
                    cs_cluster.pop(cid2)
                    cs_centroid.pop(cid2)
                    cs_deviation.pop(cid2)
                    cs_point.pop(cid2)

        #step 13 last iteration
        if i == 4:
            merged_cs_ds_clusters = dict()
            for cid1, cid2 in itertools.product(cs_cluster.keys(), ds_cluster.keys()):
                if cid1 != cid2:
                    mahalanobis_dis = mahalanobis_cluster_cluster(cs_centroid[cid1], ds_centroid[cid2], cs_deviation[cid1], ds_deviation[cid2])                        
                    if mahalanobis_dis < D:
                        D = mahalanobis_dis
                        cluster = cid2
                    
                    merged_cs_ds_clusters[cid1] = cid2

            for cid1, cid2 in merged_cs_ds_clusters.items():
                if cid1 in cs_cluster and cid2 in ds_cluster:
                    if cid1 != cid2:

                        stats, centroid, deviation = merged_cluster_statistics(cs_cluster[cid1], ds_cluster[cid2])
                        
                        ds_cluster[cid2] = stats
                        ds_centroid[cid2] = centroid    
                        ds_deviation[cid2] = deviation
                        ds_point[cid2].extend(cs_point[cid1])

                        cs_cluster.pop(cid1)
                        cs_centroid.pop(cid1)
                        cs_deviation.pop(cid1)
                        cs_point.pop(cid1)
        
        num_c, num_d = calculate_stats(cs_cluster, ds_cluster)

        result_str += 'Round ' + str(i+1) + ': ' + str(num_d) + ',' + str(len(cs_cluster)) + ',' + str(num_c) + ',' + str(len(rs)) + '\n'
    
    rs = set(int(n) for n in npdata[4][list(rs), 0])

    result = {point: cid for cid, points in ds_point.items() for point in points}
    result.update({point: -1 for cid, points in cs_point.items() for point in points})
    result.update({point: -1 for point in rs})

    write_csv(output_file, result_str, result)