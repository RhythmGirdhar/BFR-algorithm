import sys
import numpy as np
from collections import defaultdict
from sklearn.cluster import KMeans

def write_csv(output_file, result_str, result):
    with open(output_file, "w") as f:
        result_str += "\n"
        result_str += "The clustering results:\n"
        f.write(result_str)
        for point in sorted(result.keys()):
            f.write(f"{point},{result[point]}\n")
    return

def create_cluster_dict(labels):
    cluster_dict = defaultdict(list)
    
    for idx, c_id in enumerate(labels):
        cluster_dict[c_id].append(idx)

    return cluster_dict

def get_cluster_statistics(data, clusters):
    cluster = defaultdict(list)
    point_dict = defaultdict(list)
    centroid_dict = defaultdict(list)
    deviation_dict = defaultdict(list)

    for cid, idx in clusters.items():
        feature = data[idx, 2:]
        N = len(feature)
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



if __name__ == "__main__":

    # input_file = sys.argv[1]
    # n_clusters = int(sys.argv[2])
    # output_file = sys.argv[3]

    input_file = "BFR-algorithm/data/hw6_clustering.txt"
    n_clusters = 10
    output_file = "BFR-algorithm/result/task.csv"

    NO_OF_CHUNKS = 5
    
    np.random.seed(8)

    npdata = np.genfromtxt(input_file, delimiter=',')

    # Step 1. Load 20% of the data randomly.

    np.random.shuffle(npdata)
    npdata = np.array_split(npdata, NO_OF_CHUNKS)

    first_chunk = npdata[0]

    d = first_chunk.shape[1] - 2

    # Step 2. Run K-Means with a large K 

    K = 5 * n_clusters

    features = first_chunk[:, 2:]

    k_means_1 = KMeans(n_clusters = K, n_init = 10).fit(features)

    # Step 3. Move all the clusters that contain only one point to RS

    cluster_dict = create_cluster_dict(k_means_1.labels_)
    
    rs = set([idx[0] for idx in cluster_dict.values() if len(idx) == 1])

    with_rs = first_chunk[list(rs), :]

    without_rs = np.delete(first_chunk, list(rs), axis = 0)

    # Step 4. Run K-Means again to cluster the rest of the data points with K = the number of input clusters
    
    K = n_clusters

    features = without_rs[:, 2:]

    k_means_2 = KMeans(n_clusters = K, n_init = 10).fit(features)

    # Step 5. Use the K-Means result from Step 4 to generate the DS clusters

    cluster_dict = create_cluster_dict(k_means_2.labels_)

    ds_cluster, ds_point, ds_centroid, ds_deviation = get_cluster_statistics(without_rs, cluster_dict)

#     Step 6. Run K-Means on the points in the RS with a large K (e.g., 5 times of the number of the input
# clusters) to generate CS (clusters with more than one points) and RS (clusters with only one point).

    K = 5 * n_clusters

    cs_cluster = defaultdict(list)
    cs_point = defaultdict(list)
    cs_centroid = defaultdict(list)
    cs_deviation = defaultdict(list)

    if len(rs) >= K :

        features = with_rs[:, 2:]
        k_means_3 = KMeans(n_clusters = K, n_init = 10).fit(features)

        cluster_dict = create_cluster_dict(k_means_3.labels_)

        rs = set([idx[0] for idx in cluster_dict.values() if len(idx) == 1])

        cs_cluster, cs_point, cs_centroid, cs_deviation = get_cluster_statistics(with_rs, cluster_dict)

    num_d = 0
    num_c = 0

    for value in ds_cluster.values():
        num_d += value[0]

    for value in cs_cluster.values():
        num_c += value[0]

    result_str = "The intermediate results:\n"
    result_str += 'Round 1: ' + str(num_d) + ',' + str(len(cs_cluster)) + ',' + str(num_c) + ',' + str(len(rs)) + '\n'

    #TODO: WRITE TO FILE

    #Step 7 - 12

    D = 2 * np.sqrt(d)

    # for i in range(1,5):
    #     chunk = npdata[i]

    #     for idx, value in enumerate(chunk):
    #         chunk_data = value[2:]

    #         max_distance = float('inf')
    #         cluster = -1

    #         for cid, values in ds_cluster.items():
    #             mahanabolis_distance = np.sqrt(np.sum(np.square(np.divide(np.subtract(chunk_data, ds_centroid[cid]), ds_deviation[cid])), axis=0))
    #             if mahanabolis_distance < max_distance:
    #                 max_distance = mahanabolis_distance
    #                 cluster = cid

    #     print("Maxx 1: ", max_distance)
    #     print("cluster 1: ", cluster)

    #     print("================")

    
    for i in range(1, 5):
        #Step 7
        chunk = npdata[i]

        for idx, value in enumerate(chunk):
            data = value[2:]

            #Step 8
            distances = np.sqrt(np.sum(np.square(np.divide(data - list(ds_centroid.values()), list(ds_deviation.values()))), axis=1))
            cluster = np.argmin(distances)
            maxx = distances[cluster]
            cid = list(ds_cluster.keys())[cluster]
        
            if maxx < D and cid != -1:
                ds_cluster, ds_point, ds_centroid, ds_deviation = update_cluster_statistics(data, cid, value, ds_cluster, ds_point, ds_centroid, ds_deviation)

            #Step 9
            else:
                maxx = float('inf')
                cid = -1
                for c_id, values in cs_cluster.items():
                    cs_deviation_nonzero = [dev for dev in cs_deviation[c_id] if dev != 0]
                    if len(cs_deviation_nonzero) == 0:
                        # skip the calculation and assign a high distance value
                        mahalanobis_dis = np.inf
                    else:
                        mahalanobis_dis = np.sqrt(np.sum(np.square(np.divide(np.subtract(data, cs_centroid[c_id]), cs_deviation[c_id])), axis=0))
                    if mahalanobis_dis < maxx:
                        maxx = mahalanobis_dis
                        cid = c_id

                if maxx < D and cid != -1:
                    cs_cluster, cs_point, cs_centroid, cs_deviation = update_cluster_statistics(data, cid, value, cs_cluster, cs_point, cs_centroid, cs_deviation)
                
                #Step 10
                else:
                    rs.add(idx)
        
        #Step 11
        data_rs = npdata[i][list(rs), :]

        K = 5 * n_clusters

        cs_cluster = defaultdict(list)

        if len(rs) >= K :

            features = data_rs[:, 2:]
            k_means_4 = KMeans(n_clusters = K, n_init = 10).fit(features)

            cluster_dict = create_cluster_dict(k_means_4.labels_)

            rs = set([idx[0] for idx in cluster_dict.values() if len(idx) == 1])

            cs_cluster, cs_point, cs_centroid, cs_deviation = get_cluster_statistics(data_rs, cluster_dict)
        
        #Step 12
        combined_clusters = dict()
        
        for cid1 in cs_cluster.keys():
            cluster = -1
            for cid2 in cs_cluster.keys():
                if cid1 != cid2:
                    mahalanobis_dis1 = np.sqrt(np.sum(np.square(np.divide(np.subtract(cs_centroid[cid1], cs_centroid[cid2]), cs_deviation[cid2], out=np.zeros_like(np.subtract(cs_centroid[cid1], cs_centroid[cid2])), where=cs_deviation[cid2] != 0)), axis=0))
                    mahalanobis_dis2 = np.sqrt(np.sum(np.square(np.divide(np.subtract(cs_centroid[cid2], cs_centroid[cid1]), cs_deviation[cid1], out=np.zeros_like(np.subtract(cs_centroid[cid2], cs_centroid[cid1])), where=cs_deviation[cid1] != 0)), axis=0))
                    mahalanobis_dis = min(mahalanobis_dis1, mahalanobis_dis2)

                    if mahalanobis_dis < D:
                        D = mahalanobis_dis
                        cluster = cid2
            
            combined_clusters[cid1] = cluster
        
        for cid1, cid2 in combined_clusters.items():
            if cid1 in cs_cluster and cid2 in cs_cluster:
                if cid1 != cid2:
                    N = cs_cluster[cid1][0] + cs_cluster[cid2][0]
                    SUM = np.add(cs_cluster[cid1][1], cs_cluster[cid2][1])
                    SUMSQ = np.add(cs_cluster[cid1][2], cs_cluster[cid2][2])

                    centroid = SUM / N
                    deviation = np.sqrt(np.subtract(SUMSQ / N, np.square(centroid)))

                    cs_cluster[cid2] = [N, SUM, SUMSQ]
                    cs_point[cid2].extend(cs_point[cid1])
                    cs_centroid[cid2] = centroid
                    cs_deviation[cid2] = deviation

                    cs_cluster.pop(cid2)
                    cs_centroid.pop(cid2)
                    cs_point.pop(cid2)
                    cs_deviation.pop(cid2)

        #Step 13

        if i == 4:
            combined_clusters = dict()

            for cid1 in cs_cluster.keys():
                cluster = -1
                for cid2 in cs_cluster.keys():
                    if cid1 != cid2:
                        if cid2 in ds_deviation:
                            mahalanobis_dis1 = np.sqrt(np.sum(np.square(np.divide(np.subtract(cs_centroid[cid1], ds_centroid[cid2]), ds_deviation[cid2], out=np.zeros_like(np.subtract(cs_centroid[cid1], ds_centroid[cid2])), where=ds_deviation[cid2] != 0)), axis=0))
                            mahalanobis_dis2 = np.sqrt(np.sum(np.square(np.divide(np.subtract(ds_centroid[cid2], cs_centroid[cid1]), cs_deviation[cid1], out=np.zeros_like(np.subtract(ds_centroid[cid2], cs_centroid[cid1])), where=cs_deviation[cid1] != 0)), axis=0))
                            mahalanobis_dis = min(mahalanobis_dis1, mahalanobis_dis2)

                        if mahalanobis_dis < D:
                            D = mahalanobis_dis
                            cluster = cid2

                combined_clusters[cid1] = cluster
        
        for cid1, cid2 in combined_clusters.items():
            if cid1 in cs_cluster and cid2 in ds_cluster:
                if cid1 != cid2:
                    N = cs_cluster[cid1][0] + ds_cluster[cid2][0]
                    SUM = np.add(cs_cluster[cid1][1], ds_cluster[cid2][1])
                    SUMSQ = np.add(cs_cluster[cid1][2], ds_cluster[cid2][2])

                    centroid = SUM / N
                    deviation = np.sqrt(np.subtract(SUMSQ / N, np.square(centroid)))

                    ds_cluster[cid2] = [N, SUM, SUMSQ]
                    ds_point[cid2].extend(cs_point[cid1])
                    ds_centroid[cid2] = centroid
                    ds_deviation[cid2] = deviation

                    cs_cluster.pop(cid1)
                    cs_centroid.pop(cid1)
                    cs_point.pop(cid1)
                    cs_deviation.pop(cid1)
        
        num_ds = 0
        num_cs = 0
        for value in ds_cluster.values():
            num_ds += value[0]
        for value in cs_cluster.values():
            num_cs += value[0]

        result_str += 'Round ' + str(i+1) + ': ' + str(num_ds) + ',' + str(len(cs_cluster)) + ',' + str(num_cs) + ',' + str(len(rs)) + '\n'
    
    
    if len(rs) > 0:
        data_rs = npdata[4][list(rs), 0]
        rs = set([int(n) for n in data_rs])

    result = {point: cid for cid, points in ds_point.items() for point in points}
    result.update({point: -1 for cid, points in cs_point.items() for point in points})
    result.update({point: -1 for point in rs})

    write_csv(output_file, result_str, result)