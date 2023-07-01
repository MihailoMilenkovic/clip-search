import numpy as np

class KDNode:
    def __init__(self, point, embedding, left=None, right=None):
        self.point = point
        self.image_embedding = embedding
        self.left = left
        self.right = right

def build_tree(points, embeddings, depth=0):
    if not points:
        return None
    k = len(points[0])
    axis = depth % k
    sorted_points = sorted(points, key=lambda x: x[axis])
    sorted_embeddings = [embeddings[i] for i in sorted(range(len(points)), key=lambda x: points[x][axis])]
    median = len(sorted_points) // 2

    return KDNode(
        point=sorted_points[median],
        embedding=sorted_embeddings[median],
        left=build_tree(sorted_points[:median], sorted_embeddings[:median], depth + 1),
        right=build_tree(sorted_points[median + 1:], sorted_embeddings[median + 1:], depth + 1)
    )

#L2 norma za racunanje didtance
def euclidian_distance(target, node_embedd):
    return np.linalg.norm(target - node_embedd)

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