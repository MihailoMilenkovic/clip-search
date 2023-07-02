import numpy as np

class KDNode:
    def __init__(self,embedding, left=None, right=None):
        self.image_embedding = embedding
        self.left = left
        self.right = right

def build_tree(embeddings, depth=0):
    if len(embeddings) == 0:
        return None

    k = len(embeddings[0])
    axis = depth % k

    sorted_points = sorted(embeddings, key=lambda x: x[axis])
    median_idx = len(embeddings) // 2
    median = sorted_points[median_idx]

    left_points = sorted_points[:median_idx]
    right_points = sorted_points[median_idx + 1:]

    return KDNode(
        median,
        build_tree(left_points, depth + 1),
        build_tree(right_points, depth + 1)
    )

#knn with KD-tree
def k_nn(root, target, k):
    def traverse(node, target, k, depth=0, nearest_neighbors=[]):
        if node is None:
            return

        axis = depth % len(target)
        if target[axis] < node.point[axis]:
            nearer_node = node.left
            further_node = node.right
        else:
            nearer_node = node.right
            further_node = node.left

        traverse(nearer_node, target, k, depth + 1, nearest_neighbors)

        if len(nearest_neighbors) < k:
            nearest_neighbors.append(node)
        elif euclidian_distance(target, node.image_embedding) < euclidian_distance(target, nearest_neighbors[0].image_embedding):
            nearest_neighbors[0] = node
        else:
            return

        nearest_neighbors.sort(key=lambda x: euclidian_distance(target, x.image_embedding))

        if abs(target[axis] - node.point[axis]) < euclidian_distance(target, nearest_neighbors[0].image_embedding):
            traverse(further_node, target, k, depth + 1, nearest_neighbors)

    nearest_neighbors = []
    traverse(root, target, k, depth=0, nearest_neighbors=nearest_neighbors)
    return nearest_neighbors

#L2 norma za racunanje didtance
def euclidian_distance(target, node_embedd):
    return np.linalg.norm(target - node_embedd)

def knn_algorithm(query_point, kdTree, k):
    knn_array = knn_search(query_point, kdTree, k)
    print(knn_array)

def knn_search(query_point, kdtree, k):
    k_nearest_neighbors = []

    def search(node, depth=0):
        nonlocal k_nearest_neighbors

        if node is None:
            return

        axis = depth % len(query_point)
        current_point = node.point
        distance = euclidian_distance(query_point - current_point)

        if len(k_nearest_neighbors) < k:
            k_nearest_neighbors.append((current_point, distance))
            k_nearest_neighbors = sorted(k_nearest_neighbors, key=lambda x: x[1])
        else:
            if distance < k_nearest_neighbors[-1][1]:
                k_nearest_neighbors[-1] = (current_point, distance)
                k_nearest_neighbors = sorted(k_nearest_neighbors, key=lambda x: x[1])

        if query_point[axis] < current_point[axis]:
            search(node.left, depth + 1)
            if abs(query_point[axis] - current_point[axis]) < k_nearest_neighbors[-1][1]:
                search(node.right, depth + 1)
        else:
            search(node.right, depth + 1)
            if abs(query_point[axis] - current_point[axis]) < k_nearest_neighbors[-1][1]:
                search(node.left, depth + 1)

    search(kdtree)
    return [neighbor for neighbor, _ in k_nearest_neighbors]