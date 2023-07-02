import numpy as np

#knn using cosine similarity for metric
def calculate_cosine_similarity(query_embedd, curr_image):
    dot_product = np.dot(query_embedd, curr_image)
    similarity = dot_product / (np.linalg.norm(query_embedd)* np.linalg.norm(curr_image))
    return similarity

def knn_search_cosine(query_embedd,tree, k):
    k_nearest_neighbors = []

    def search(node, depth=0):
        nonlocal k_nearest_neighbors

        if node is None:
            return

        dimension = depth % len(query_embedd) 
        current_embedd = node.image_embedding

        similarity = calculate_cosine_similarity(query_embedd, current_embedd)

        if len(k_nearest_neighbors) < k or similarity > k_nearest_neighbors[-1][1]:
            k_nearest_neighbors.append((current_embedd, similarity))
            k_nearest_neighbors = sorted(k_nearest_neighbors, key=lambda x: x[1], reverse=True)
            k_nearest_neighbors = k_nearest_neighbors[:k]

        if np.all(query_embedd[dimension] < current_embedd[dimension]):
            search(node.left, depth+1)
        else:
            search(node.right, depth+1)

    search(tree)

    return k_nearest_neighbors[0][0]

def knn_algorithm_euclidian(query_embedd, kdTree, k):
    image_embedd = knn_search(query_embedd, kdTree, k)
    return image_embedd

#L2 norm for computing distance
def euclidian_distance(target, node_embedd):
    return np.linalg.norm(target - node_embedd)

#Knn search of kd tree
def knn_search(query_embedd, kdtree, k):
    k_nearest_neighbors = []

    def search(node, depth=0):
        nonlocal k_nearest_neighbors

        if node is None:
            return

        axis = depth % len(query_embedd) 
        current_embedd = node.image_embedding #ndarray, as query_embedd
        distance = euclidian_distance(query_embedd, current_embedd)

        if len(k_nearest_neighbors) < k:
            k_nearest_neighbors.append((current_embedd, distance))
            k_nearest_neighbors = sorted(k_nearest_neighbors, key=lambda x: x[1])
        else:
            if distance < k_nearest_neighbors[-1][1]:
                k_nearest_neighbors[-1] = (current_embedd, distance)
                k_nearest_neighbors = sorted(k_nearest_neighbors, key=lambda x: x[1])

        if np.all(query_embedd[axis] < current_embedd[axis]):
            search(node.left, depth + 1)
            if np.all(abs(query_embedd[axis] - current_embedd[axis])) < k_nearest_neighbors[-1][1]:
                search(node.right, depth + 1)
        else:
            search(node.right, depth + 1)
            if np.all(abs(query_embedd[axis] - current_embedd[axis])) < k_nearest_neighbors[-1][1]:
                search(node.left, depth + 1)

    search(kdtree)
    return k_nearest_neighbors[0][0] #return the closest image embedding (it is sorted)

#Simple iterative kNN, can be tested if desired
def get_k_most_similar_images_L2(query_embedding, image_embeddings, k=1):
    computed_distances = []
    # Computed distances by Euclidian (L2) distance
    for i, curr_embedding in enumerate(image_embeddings):
        distance = euclidian_distance(query_embedding, curr_embedding)
        computed_distances.append((curr_embedding, distance))
    
    # Sort image embeddings according to distance
    computed_distances.sort(key=lambda x: x[1])
    
    # K neairest embeddings
    image_embedding = [embedding for embedding, _ in computed_distances[:k]] 
    return image_embedding