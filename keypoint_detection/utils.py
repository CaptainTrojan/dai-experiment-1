class KeypointSmoother:
    def __init__(self):
        self.keypoint_history = []
        self.keypoint_history_length = 5
        
    def smooth_keypoints(self, keypoints):
        if len(self.keypoint_history) < self.keypoint_history_length:
            self.keypoint_history.append(keypoints)
            return keypoints
        else:
            self.keypoint_history.pop(0)
            self.keypoint_history.append(keypoints)
            return self.average_keypoints()
        
    def average_keypoints(self):
        averaged_keypoints = []
        for i in range(len(self.keypoint_history[0])):
            averaged_keypoints.append(self.average_keypoint(i))
        return averaged_keypoints
    
    def average_keypoint(self, idx):
        x_sum = 0
        y_sum = 0
        count = 0
        
        # if newest is none, return none
        if self.keypoint_history[-1][idx] is None:
            return None
        
        for keypoints in self.keypoint_history:
            if keypoints[idx] is not None:
                x_sum += keypoints[idx][0]
                y_sum += keypoints[idx][1]
                count += 1
        if count == 0:
            return None
        else:
            return (x_sum / count, y_sum / count)
    
    
class KeypointGraph:   
    CP = [
        (192, 64, 64),    # Lighter Red
        (64, 192, 64),    # Lighter Green
        (64, 64, 192),    # Lighter Blue
        (192, 192, 64),   # Lighter Yellow
        (192, 64, 192),   # Lighter Magenta
        (64, 192, 192),   # Lighter Cyan
        (192, 123, 64),   # Lighter Orange
        (96, 64, 96)      # Lighter Purple
    ]

    # full
    # POSE_PAIRS = [[1, 2], [1, 5], [2, 3], [3, 4], [5, 6], [6, 7], [1, 8], [8, 9], [9, 10], [1, 11], [11, 12], [12, 13],
    #      [1, 0], [0, 14], [14, 16], [0, 15], [15, 17], [5, 17], [2, 16]]
    
    # disconnected limbs and face features
    POSE_PAIRS = [
        (1, 2), (1, 5), # neck -> shoulders
        (2, 3), (3, 4), # right arm
        (5, 6), (6, 7), # left arm
        (1, 8), (8, 9), (9, 10), # right leg
        (1, 11), (11, 12), (12, 13), # left leg
        (0, 1), # neck
        (0, 14), (14, 16), # right face
        (0, 15), (15, 17)  # left face
    ] 
    
    edge_colors = [
        CP[1], CP[2], 
        CP[1], CP[1],
        CP[2], CP[2],
        CP[3], CP[3], CP[3],
        CP[4], CP[4], CP[4],
        CP[5],
        CP[6], CP[6],
        CP[7], CP[7]
    ]
    
    edge_to_color = {
        k: v for k, v in zip(POSE_PAIRS, edge_colors)
    }

    @staticmethod
    def generate_continuous_graph(keypoints):
        list_of_neighbors = KeypointGraph.build_full_graph(keypoints)
        vertices, edges = KeypointGraph.get_neck_component(list_of_neighbors)
        return vertices, edges
    
    @staticmethod
    def build_full_graph(keypoints):
        list_of_neighbors = []
        for i in range(len(keypoints)):
            list_of_neighbors.append([])
        
        for edge in KeypointGraph.POSE_PAIRS:
            if keypoints[edge[0]] is not None and keypoints[edge[1]] is not None:
                list_of_neighbors[edge[0]].append(edge[1])
                list_of_neighbors[edge[1]].append(edge[0])
        return list_of_neighbors
    
    @staticmethod
    def get_neck_component(list_of_neighbors):
        if len(list_of_neighbors[1]) == 0:
            return [], []
        
        edges = []
        visited = set()
        def dfs(node):
            if node in visited:
                return
            visited.add(node)
            for neighbor in list_of_neighbors[node]:
                a = min(node, neighbor)
                b = max(node, neighbor)
                edges.append((a, b, KeypointGraph.edge_to_color[(a, b)]))
                dfs(neighbor)
        dfs(1)
        return [(v, (0, 0, 0)) for v in visited], edges