# Pathplanning - A* core logic

import numpy as np
import cv2
import heapq
import matplotlib.pyplot as plt

# Coordinates are ALWAYS (x, y, z)
# But in heightmap : (row = y, col = x)

class PathPlannerCore:
    def __init__(self, heightmap_path, start, goal, z_offset, height_scale):
        self.heightmap_path = heightmap_path
        self.start = start
        self.goal = goal
        self.z_offset = z_offset
        self.height_scale = height_scale

        self.heightmap_uint16 = self.load_heightmap(heightmap_path)
        self.H, self.W = self.heightmap_uint16.shape

        self.heightmap = self.heightmap_uint16 * height_scale

    def load_heightmap(self, path):
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise RuntimeError(f"[load_heightmap] Failed to load: {path}")
        
        img = img.astype(np.float32)

        vmin, vmax = img.min(), img.max()

        img_norm = (img - vmin) / (vmax - vmin)
        img_uint16 = 65535 * img_norm
        img_uint16 = img_uint16.astype(np.uint16)

        return img_uint16

    # ----------------
    # A* core 함수
    # ----------------
    def astar(self, start, goal):
        H, W = self.H, self.W
        heightmap = self.heightmap

        # -------- 8 방향 이동 정의 --------
        # moves = (x, y, range)
        # 좌 우 하 상 좌하 좌상 우하 우상수
        moves = [
            ( 1,  0, 1.0),
            (-1,  0, 1.0),
            ( 0,  1, 1.0),
            ( 0, -1, 1.0),
            ( 1,  1, 1.41421356),
            ( 1, -1, 1.41421356),
            (-1,  1, 1.41421356),
            (-1, -1, 1.41421356),
        ]

        # ----- 휴리스틱: goal까지 직선거리 -----
        def h(a, b):
            return ((a[0] - b[0])**2 + (a[1] - b[1])**2)**0.5

        # 우선순위 큐: (f, g, (x,y))
        open_set = []
        heapq.heappush(open_set, (0, 0, start))

        came_from = {}
        g = {start: 0}

        while open_set:
            f, cost, node = heapq.heappop(open_set)
            x, y = node

            if node == goal:
                # reconstruct path
                path = [node]
                while node in came_from:
                    node = came_from[node]
                    path.append(node)
                return path[::-1]

            curr_h = heightmap[y, x]

            # -------- 8방향 탐색 ---------
            for dx, dy, base_cost in moves:
                nx, ny = x + dx, y + dy

                if not (0 <= nx < W and 0 <= ny < H):
                    continue

                next_h = heightmap[ny, nx]

                # ---------- Edge Cost 계산 ----------
                elev_diff = abs(float(next_h) - float(curr_h))
                edge_cost = (base_cost**2 + elev_diff**2)**0.5

                new_g = cost + edge_cost

                # g값 갱신 필요 여부 확인
                if (nx, ny) not in g or new_g < g[(nx, ny)]:
                    g[(nx, ny)] = new_g
                    came_from[(nx, ny)] = (x, y)
                    f = new_g + h((nx, ny), goal)
                    heapq.heappush(open_set, (f, new_g, (nx, ny)))

        return None

    # -------------------
    # 전체 경로 계획 wrapper
    # -------------------
    def plan(self):
        start = (int(self.start[0]), int(self.start[1]))
        goal  = (int(self.goal[0]),  int(self.goal[1]))

        path = self.astar(start, goal)
        if path is None:
            raise RuntimeError("A* path not found")

        # scale back to meters
        path_x = [p[0] for p in path]
        path_y = [p[1] for p in path]
        path_z = [self.heightmap[p[1],p[0]] * self.height_scale + self.z_offset for p in path] # row(y), col(x)

        return path_x, path_y, path_z

    # -------------------
    # Visualization
    # -------------------
    def plot_path_2d(self, x, y, save=None):
        plt.imshow(self.heightmap, cmap="gray")
        plt.plot(x, y, 'r-')
        if save:
            plt.savefig(save)

    def plot_path_3d(self, x, y, z, save=None):
        pass



if __name__ == "__main__":

    heightmap_path = "map/1024-001.png"

    height_scale  = 1000/65535
    height_offset = 200
    start = (100, 100, 10)
    goal  = (1000, 1000, 10)

    print("[TEST] : Running PathPlannerCore")

    planner = PathPlannerCore(heightmap_path, start, goal, height_offset, height_scale)

    path_x, path_y, path_z = planner.plan()

    print("Generated Path Points:")

    for i in range(len(path_x)):
        print(f"{i}: ({path_x[i]:.1f}, {path_y[i]:.1f}, {path_z[i]:.1f})")

    planner.plot_path_2d(path_x, path_y, save="test_path_2d.png")
    print("2D plot saved as test_path_2d.png")