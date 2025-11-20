from Plan2WP_core import PathPlannerCore
import numpy as np

heightmap_path = "map/1000-001.png"
start = (0,0,0)
goal  = (0,0,0)
z_offset = 0.1
height_scale = 1000/65535

test = PathPlannerCore(heightmap_path, start, goal, z_offset, height_scale)

map = test.heightmap
map_max = np.max(map)
map_min = np.min(map)

print(map_max, map_min )