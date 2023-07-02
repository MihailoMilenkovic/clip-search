class KDNode:
    def __init__(self,embedding, left=None, right=None):
        self.image_embedding = embedding
        self.left = left
        self.right = right

#build KD tree of image embeddings
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

#print KD tree
def print_kd_tree(node, depth=0):
    if node is None:
        return

    k = len(node.image_embedding)
    axis = depth % k

    print("  " * depth, end="")
    print(f"- {node.image_embedding}")

    print_kd_tree(node.left, depth + 1)
    print_kd_tree(node.right, depth + 1)
