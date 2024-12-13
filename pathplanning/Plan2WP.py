# Copyright 2016 Open Source Robotics Foundation, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import rclpy
from rclpy.node import Node
import numpy as np
import onnx
import onnxruntime as ort
import cv2
from gymnasium import spaces
import torch.nn.functional as F
import networkx as nx
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

#############################################################################################################
# added by controller
from custom_msgs.msg import GlobalWaypointSetpoint, LocalWaypointSetpoint
from custom_msgs.msg import Heartbeat


#############################################################################################################
class PathPlanning:
    def __init__(self, onnx_path, heightmap_path, start, goal, n_waypoints=8, scale_factor=10, target_size=80,
                 z_factor=0.5):
        self.onnx_path = onnx_path
        self.heightmap_path = heightmap_path
        self.start = start
        self.goal = goal

        #start_arr = np.array(self.start, dtype=float)
        #print('+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
        #print(start[0])
        #print(int(start[0]))
        #print('+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
        #print(np.array(start))
        #print(self.start_x)
        
        self.n_waypoints = n_waypoints
        self.scale_factor = scale_factor
        self.target_size = target_size
        self.z_factor = z_factor  # New z_factor attribute
        # Load and preprocess heightmap
        self.heightmap = self.load_heightmap(heightmap_path)
        self.heightmap_resized = self.resize_heightmap(self.heightmap, target_size)
        self.h, self.w = self.heightmap_resized.shape
        self.distance = min(self.h, self.w) * 2 / 3
        self.h_origin, self.w_origin = self.heightmap.shape
        self.scale_factor_waypoint_h = self.h_origin/self.h # Scale Factor of waypoint
        self.scale_factor_waypoint_w = self.w_origin/self.w # Scale Factor of waypoint

        # Check distance between start and goal
        if np.linalg.norm(np.array(start) - np.array(goal)) < self.distance:
            raise ValueError("Start and Goal is too close")


    def load_heightmap(self, path):
        heightmap_image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        heightmap = cv2.normalize(heightmap_image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        if heightmap is None:
            raise ValueError(f"Failed to load heightmap from path: {path}")
        return heightmap

    def resize_heightmap(self, heightmap, target_size):
        resize_factor = max(heightmap.shape) // target_size
        resize_factor = max(resize_factor, 1)
        heightmap_resized = cv2.resize(heightmap,
                                       (heightmap.shape[1] // resize_factor, heightmap.shape[0] // resize_factor)) #scale
        return heightmap_resized

  # 수정된 _get_obs 함수 (시작점과 도착점 표시 명확화)
    def _get_obs(self):
        # 높이맵 정규화 (0 to 255)
        height_normalized = cv2.normalize(self.heightmap_resized, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        
        # 경사도 계산 및 정규화
        gradient_x = cv2.Sobel(self.heightmap_resized, cv2.CV_64F, 1, 0, ksize=3)
        gradient_y = cv2.Sobel(self.heightmap_resized, cv2.CV_64F, 0, 1, ksize=3)
        gradient_magnitude = np.sqrt(gradient_x**2 + gradient_y**2)
        gradient_normalized = cv2.normalize(gradient_magnitude, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        
        # 곡률 계산 및 정규화
        curvature = cv2.Laplacian(self.heightmap_resized, cv2.CV_64F)
        curvature_normalized = cv2.normalize(curvature, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        
        # 3채널 이미지 생성
        self.raw_obs = np.stack([height_normalized, gradient_normalized, curvature_normalized], axis=-1)
        
        # 경로 강조를 위한 마스크 생성
        path_mask = np.zeros_like(height_normalized)
        
        # 시작점과 목표점 표시
        #start_x, start_y = self.start
        #goal_x, goal_y = self.goal
        start_x = int(self.start[0]/self.scale_factor_waypoint_w)
        start_y = int(self.start[1]/self.scale_factor_waypoint_h)
        
        goal_x = int(self.goal[0]/self.scale_factor_waypoint_w)
        goal_y = int(self.goal[1]/self.scale_factor_waypoint_h)
       
        
        cv2.circle(path_mask, (start_y, start_x), 3, 255, -1)
        cv2.circle(path_mask, (goal_y, goal_x), 3, 255, -1)
        
        # 경로 표시 (고도에 따라 색상 변화)
        # if len(self.agent1_path) > 1:
        #     for i in range(len(self.agent1_path) - 1):
        #         x1, y1 = self.agent1_path[i]
        #         x2, y2 = self.agent1_path[i+1]
                
        #         # 현재 세그먼트의 평균 고도 계산
        #         avg_height = (self.heightmap_resized[x1, y1] + self.heightmap_resized[x2, y2]) / 2
                
        #         # 고도에 따라 색상 결정 (낮은 고도: 얇은 선, 높은 고도: 두꺼운 선)
        #         thickness = int(1 + (avg_height / np.max(self.heightmap_resized)) * 4)
        #         cv2.line(path_mask, (y1, x1), (y2, x2), 255, thickness)
        
        # 경로 마스크를 이용해 원본 이미지에 경로 강조
        self.raw_obs[:,:,0] = cv2.addWeighted(self.raw_obs[:,:,0], 1, path_mask, 0.5, 0)
        
        # 채널 순서 변경 (H, W, C) -> (C, H, W)
        obs = np.transpose(self.raw_obs, (2, 0, 1))
        
        # Resize observation to match the defined observation space
        obs = cv2.resize(np.transpose(obs, (1, 2, 0)), (80, 80), interpolation=cv2.INTER_AREA)
        obs = np.transpose(obs, (2, 0, 1))
        obs = np.expand_dims(obs, axis=0).astype(np.float32)
        
        return obs
    
    # def extract_features(self, obs):
    #     obs_resized = cv2.resize(obs, (80, 80))

    #     obs_tensor = np.transpose(obs_resized, (2, 0, 1))
    #     obs_tensor = np.expand_dims(obs_tensor, axis=0)

    #     obs_tensor = obs_tensor.astype(np.float32) / 255.0

    #     return obs_tensor

    def plan_path(self, init, target):
        
        ### path planning by learning ###
        # Observation = self._get_obs()

        # start = (int(init[0]/self.scale_factor_waypoint_w), int(init[2]/self.scale_factor_waypoint_h))
        # goal = (int(target[0]/self.scale_factor_waypoint_w), int(target[2]/self.scale_factor_waypoint_h))

        # start_z = init[1]
        # goal_z = target[1]

        # ort_session = ort.InferenceSession(self.onnx_path)
        # action = ort_session.run(None, {"observation": Observation})
        # action = np.clip(action, -1, 1)

        # direction_vector = np.array(goal) - np.array(start)
        # direction_vector = direction_vector / np.linalg.norm(direction_vector)
        # perpendicular_vector = np.array([-direction_vector[1], direction_vector[0]])

        # waypoints = np.linspace(start, goal, self.n_waypoints)

        # for i in range(len(waypoints)):
        #     waypoints[i] += action[0][0][i] * perpendicular_vector * self.scale_factor
        #     waypoints[i] = np.clip(waypoints[i], [0, 0],
        #                            [self.heightmap_resized.shape[0] - 1, self.heightmap_resized.shape[1] - 1])

        # agent1_path = waypoints.astype(int).tolist()
        # dijkstra_path = self.find_shortest_path(agent1_path)
        # cnn_path = [start] + dijkstra_path + [goal]
        # cnn_real_path = [start] + agent1_path + [goal]

        # cnn_path = np.array(cnn_path)
        # cnn_real_path = np.array(cnn_real_path)

        # # Calculate z values based on heightmap
        # #path_z = np.array([self.heightmap_resized[int(point[0]), int(point[1])] for point in cnn_real_path])
        # # agent1_path_z = np.array([self.heightmap_resized[int(point[0]), int(point[1])] for point in agent1_path]) + self.z_factor
        # for i in range(len(agent1_path)):
        #     agent1_path[i][0] = agent1_path[i][0] * self.scale_factor_waypoint_w
        #     agent1_path[i][1] = agent1_path[i][1] * self.scale_factor_waypoint_h
        
        # agent1_path_z = np.array([self.heightmap[int(point[0]), int(point[1])] for point in agent1_path])*0.1 + self.z_factor
        # path_z = np.insert(agent1_path_z, 0, start_z)
        # path_z = np.append(path_z, goal_z)
        # path_z = np.float64(path_z)

        # # Apply z_factor
        # # path_z = path_z + self.z_factor


        # self.path_x_learning = cnn_real_path[:, 0]
        # self.path_y_learning = cnn_real_path[:, 1]
        # self.path_z_learning = path_z

        # self.path_x = cnn_real_path[:, 0] * self.scale_factor_waypoint_w
        # self.path_y = cnn_real_path[:, 1] * self.scale_factor_waypoint_h
        # self.path_z = path_z
        # self.path_x,self.path_y,self.path_z = self.add_waypoint_main(self.path_x,self.path_y,self.path_z,self.heightmap)

        ### path planning by pso ###
        start_pso = np.array([[init[0], init[2]]])
        goal_pso = np.array([[target[0], target[2]]])

        # gBest_value, waypoint = self.pso_single(self.heightmap*self.h_origin, start_pso, goal_pso, self.h_origin)
        gBest_value, waypoint = self.pso_single(self.heightmap*0.1, start_pso, goal_pso, self.h_origin)
        self.path_x_pso = waypoint[:,0]
        self.path_y_pso = waypoint[:,1]

        self.path_z_pso = np.array([self.heightmap[int(point[0]), int(point[1])] for point in waypoint])*0.1 + self.z_factor
        self.path_x = self.path_x_pso
        self.path_y = self.path_y_pso
        self.path_z = self.path_z_pso
        # self.path_x,self.path_y,self.path_z = self.add_waypoint_main(self.path_x_pso,self.path_y_pso,self.path_z_pso,self.heightmap*0.1)
        
        ### OUTPUT ###
        # path_final_3D_learning_model = np.column_stack((self.path_x_learning, self.path_y_learning, self.path_z_learning)) # output path of learning model scaled target size
        path_final_3D = np.column_stack((self.path_x, self.path_y, self.path_z)) # real path

        # print("Output path of learning model :",path_final_3D_learning_model)
        print("Output Real Path", path_final_3D)

        # 경로생성 결과 확인용        
        self.plot_path_2d("/home/user/workspace/ros2/ros2_ws/src/pathplanning/pathplanning/Results_Images/path_2d.png")
        self.plot_path_3d("/home/user/workspace/ros2/ros2_ws/src/pathplanning/pathplanning/Results_Images/path_3d.png")
        # self.plot_path_2d_learning("/home/user/workspace/ros2/ros2_ws/src/pathplanning/pathplanning/Results_Images/path_2d_learning.png")
        # self.plot_path_3d_learning("/home/user/workspace/ros2/ros2_ws/src/pathplanning/pathplanning/Results_Images/path_3d_learning.png")

    def pso_single(self,heightmap, Start, Goal, length_heightmap): # height map scale : 0 ~ max height(heightmap size)
        #start_time = time.time()
        num_particles = 20
        num_waypoints = 6
        num_dimensions = 1
        dimensionality = num_waypoints * num_dimensions
        max_iter = 15000
        w = 0.51
        c1 =2
        c2 =2
        
        vertical_length_st = 120 # map size : 1024-1024 -> 120 , map size : 788 -> 80
        
        height_start = heightmap[round(Start[0][0]), round(Start[0][1])]
        height_goal = heightmap[round(Goal[0][0]), round(Goal[0][1])]

        # initialize 
        x = np.random.rand(num_particles, dimensionality) * 2 - 1
        v = np.zeros((num_particles, dimensionality))
        pBest = x

        pBest_value = self.cost_cal(Start, Goal, x, num_particles, num_waypoints, heightmap, vertical_length_st, length_heightmap)
        idx = np.argmin(pBest_value)
        gBest_value = np.min(pBest_value)
        gBest = pBest[idx, :]

        for iter in range(max_iter):
            r1 = np.random.rand(num_particles, dimensionality)
            r2 = np.random.rand(num_particles, dimensionality)

            v = w * v + c1 * r1 * (pBest - x) + c2 * r2 * (gBest - x)
            x = x + v
            x = np.maximum(-1, np.minimum(1, x))

            current_value = self.cost_cal(Start, Goal, x, num_particles, num_waypoints, heightmap, vertical_length_st, length_heightmap)
            idx = current_value < pBest_value
            pBest[idx, :] = x[idx, :]
            pBest_value[idx] = current_value[idx]

            idx = np.argmin(pBest_value)
            current_min_value = np.min(pBest_value)
            if current_min_value < gBest_value:
                gBest = pBest[idx, :]
                gBest_value = np.min(pBest_value)
            
            end_condition = pBest_value - gBest_value
            if np.sum(np.abs(end_condition)) <=0.5:
                break
        #cal_time = time.time() - start_time

        # g_Best -> waypoint 
        height_Start = heightmap[round(Start[0][0]), round(Start[0][1])]
        height_Goal = heightmap[round(Goal[0][0]), round(Goal[0][1])]
        
        Start_3D = np.array([[Start[0][0], Start[0][1], height_Start]])
        Goal_3D = np.array([[Goal[0][0], Goal[0][1], height_Goal]])
        
        vec_straight_2D = Goal - Start
        vec_straight_3D = Goal_3D - Start_3D
        
        dist_straight_2D = np.linalg.norm(vec_straight_2D)
        dist_straight_3D = np.linalg.norm(vec_straight_3D)
        
        vec_vertical = np.array([[-vec_straight_2D[0][1], vec_straight_2D[0][0]]]) / dist_straight_2D
        
        
        length_vertical = gBest[:] * vertical_length_st
        standard_point = np.zeros((num_waypoints, 2))
        waypoint = np.zeros((num_waypoints, 2))
        for k in range(num_waypoints):
            standard_point[k, :] = Start + (Goal - Start) * (k / (num_waypoints + 1))
            waypoint[k,:] = standard_point[k,:] + length_vertical[k] * vec_vertical
            waypoint[k,0] = np.maximum(1, np.minimum(waypoint[k,0], length_heightmap))
            waypoint[k,1] = np.maximum(1, np.minimum(waypoint[k,1], length_heightmap))

        waypoint = np.concatenate((Start, waypoint), axis=0)
        waypoint = np.concatenate((waypoint, Goal), axis=0)

        return gBest_value, waypoint

    def cost_cal(self,Start, Goal, x, num_particles, num_waypoints, heightmap, length_vertical_st, length_heightmap):
    
        height_Start = heightmap[round(Start[0][0]), round(Start[0][1])]
        height_Goal = heightmap[round(Goal[0][0]), round(Goal[0][1])]
        
        Start_3D = np.array([[Start[0][0], Start[0][1], height_Start]])
        Goal_3D = np.array([[Goal[0][0], Goal[0][1], height_Goal]])
        
        vec_straight_2D = Goal - Start
        vec_straight_3D = Goal_3D - Start_3D
        
        dist_straight_2D = np.linalg.norm(vec_straight_2D)
        dist_straight_3D = np.linalg.norm(vec_straight_3D)
        
        vec_vertical = np.array([[-vec_straight_2D[0][1], vec_straight_2D[0][0]]]) / dist_straight_2D
        Cost_value = np.zeros((num_particles, 1))
        
        num_interp = 10
        
        for i in range(num_particles):
            length_vertical = x[i, :] * length_vertical_st
            standard_point = np.zeros((num_waypoints, 2))
            waypoint = np.zeros((num_waypoints, 2))
            for k in range(num_waypoints):
                standard_point[k, :] = Start + (Goal - Start) * (k / (num_waypoints + 1))
                waypoint[k,:] = standard_point[k,:] + length_vertical[k] * vec_vertical
                waypoint[k,0] = np.maximum(1, np.minimum(waypoint[k,0], length_heightmap))
                waypoint[k,1] = np.maximum(1, np.minimum(waypoint[k,1], length_heightmap))

            waypoint = np.concatenate((Start, waypoint), axis=0)
            waypoint = np.concatenate((waypoint, Goal), axis=0)

            all_points = Start # Start point add
            
            rows, cols = waypoint.shape
            for wp in range(rows-1):
                for j in range(num_interp):
                    t = j / (num_interp + 1)
                    interp_point = waypoint[wp,:] * (1-t) + waypoint[wp+1,:] * t
                    interp_point = interp_point.reshape(1, 2)
                    all_points = np.concatenate((all_points, interp_point), axis=0)

            all_points = np.concatenate((all_points, Goal), axis=0) # Goal point add
            rows, cols = all_points.shape
            
            dist_total = 0
            for j in range(rows-1):
                p1 = all_points[j,:]
                p2 = all_points[j+1,:]

                p1 = np.maximum(1, np.minimum(p1, length_heightmap))
                p2 = np.maximum(1, np.minimum(p2, length_heightmap))

                height1 = heightmap[round(p1[0]), round(p1[1])]
                height2 = heightmap[round(p2[0]), round(p2[1])]

                p1_3D = np.array([[p1[0], p1[1], height1]])
                p2_3D = np.array([[p2[0], p2[1], height2]])

                dist = np.linalg.norm(p2_3D - p1_3D)
                dist_total = dist_total + dist
            path_ratio = dist_total / dist_straight_3D
            Cost_value[i] = path_ratio
            Cost_value = Cost_value.reshape(-1)
            
        return Cost_value

    def add_waypoint(self,i,index,result_x,result_y,result_z,terrain_z):
        return_x = []
        return_y = []
        return_z = []
        return_index = []
        start = index[i]
        end = index[i+1]
        
        segment_x = result_x[start:end+1]
        segment_y = result_y[start:end+1]
        segment_z = result_z[start:end+1]
        segment_terrain_z = terrain_z[start:end+1]
        gap_z = segment_z - segment_terrain_z
        segment_min = gap_z.min()
        segment_index_min = gap_z.argmin()
        global_index_min = start + segment_index_min 
        segment_max = gap_z.max()
        segment_index_max = gap_z.argmax()
        global_index_max = start + segment_index_max
        if segment_max < 0:
            return_x.append(segment_x[segment_index_min])
            return_y.append(segment_y[segment_index_min])
            return_z.append(segment_terrain_z[segment_index_min]) 
            return_index.append(global_index_min)

        elif segment_min > 0:
            return_x.append(segment_x[segment_index_max]) 
            return_y.append(segment_y[segment_index_max])
            return_z.append(segment_terrain_z[segment_index_max])
            return_index.append(global_index_max)
        else:
            if global_index_min < global_index_max:
                return_x.append(segment_x[segment_index_min])
                return_y.append(segment_y[segment_index_min])
                return_z.append(segment_terrain_z[segment_index_min]) 
                return_index.append(global_index_min)
                return_x.append(segment_x[segment_index_max]) 
                return_y.append(segment_y[segment_index_max])
                return_z.append(segment_terrain_z[segment_index_max])
                return_index.append(global_index_max)
            else:
                return_x.append(segment_x[segment_index_max]) 
                return_y.append(segment_y[segment_index_max])
                return_z.append(segment_terrain_z[segment_index_max])
                return_index.append(global_index_max)
                return_x.append(segment_x[segment_index_min])
                return_y.append(segment_y[segment_index_min])
                return_z.append(segment_terrain_z[segment_index_min]) 
                return_index.append(global_index_min)
        return np.array(return_x),np.array(return_y),np.array(return_z),np.array(return_index)

   
    def add_waypoint_main(self,waypoint_x,waypoint_y,waypoint_z,heightmap):
        z = heightmap*1
        ###########Making Way by Interporlation(Not way point)##################
        num_points = len(waypoint_x)
        num_total_points = 1000
        # x, y, z 좌표를 각각 분리
        x_vals = waypoint_x
        y_vals = waypoint_y
        z_vals = waypoint_z
        
        # 각 경로 구간에서 보간할 t 값 계산 (0에서 num_points-1까지)
        t_original = np.linspace(0, num_points - 1, num_points)
        t_interpolated = np.linspace(0, num_points - 1, num_total_points)
        
        # 선형 보간 수행
        way_x = np.interp(t_interpolated, t_original, x_vals)
        way_y = np.interp(t_interpolated, t_original, y_vals)
        way_z = np.interp(t_interpolated, t_original, z_vals)
        
        # 원래 경로점들의 보간된 점에서의 인덱스 계산
        original_indices_in_interpolated = np.searchsorted(t_interpolated, t_original)
        
        #############################Get Terrain of Way#########################
        #non interp
        int_way_x = list(map(int, way_x))
        int_way_y = list(map(int, way_y))
        terrain_z = np.array(z[int_way_x, int_way_y])
        
        ############################Add Waypoint###############################
        add_x_list=[]
        add_y_list=[]
        add_z_list=[]
        index_add=[]
        return_index_list=[]
        for i in range(len(waypoint_z)-1):
            return_x,return_y,return_z,return_index = self.add_waypoint(i,original_indices_in_interpolated,way_x,way_y,way_z,terrain_z)
            add_x_list.append(return_x)
            add_y_list.append(return_y)
            add_z_list.append(return_z)
            return_index_list.append(return_index)
            index_add.append(i+1)
            
        # 삽입을 위한 작업
        new_waypoint_x = waypoint_x.copy()
        new_waypoint_z = waypoint_z.copy()  # 원본 배열을 복사하여 작업
        new_waypoint_y = waypoint_y.copy() 
        new_index = original_indices_in_interpolated.copy()
        # 삽입 작업에 따른 위치 조정
        for i, (array, pos) in enumerate(zip(add_z_list, index_add)):
            # 배열이 삽입된 후에 뒤의 위치들은 삽입된 배열의 길이만큼 증가합니다.
            adjusted_pos = pos + sum(len(add_z_list[j]) for j in range(i))
            new_waypoint_z = np.insert(new_waypoint_z, adjusted_pos, array)
        
        for i, (array, pos) in enumerate(zip(add_y_list, index_add)):
            # 배열이 삽입된 후에 뒤의 위치들은 삽입된 배열의 길이만큼 증가합니다.
            adjusted_pos = pos + sum(len(add_y_list[j]) for j in range(i))
            new_waypoint_y = np.insert(new_waypoint_y, adjusted_pos, array)
        
        for i, (array, pos) in enumerate(zip(add_x_list, index_add)):
            # 배열이 삽입된 후에 뒤의 위치들은 삽입된 배열의 길이만큼 증가합니다.
            adjusted_pos = pos + sum(len(add_x_list[j]) for j in range(i))
            new_waypoint_x = np.insert(new_waypoint_x, adjusted_pos, array)
            
        for i, (array, pos) in enumerate(zip(return_index_list, index_add)):
            # 배열이 삽입된 후에 뒤의 위치들은 삽입된 배열의 길이만큼 증가합니다.
            adjusted_pos = pos + sum(len(return_index_list[j]) for j in range(i))
            new_index = np.insert(new_index, adjusted_pos, array)

        return new_waypoint_x, new_waypoint_y,new_waypoint_z

    def add_waypoint_main_2(self,waypoint_x,waypoint_y,waypoint_z,heightmap):
        z = heightmap*1
        ###########Making Way by Interporlation(Not way point)##################
        num_points = len(waypoint_x)
        num_total_points = 1000
        # x, y, z 좌표를 각각 분리
        x_vals = waypoint_x
        y_vals = waypoint_y
        z_vals = waypoint_z
        
        # 각 경로 구간에서 보간할 t 값 계산 (0에서 num_points-1까지)
        t_original = np.linspace(0, num_points - 1, num_points)
        t_interpolated = np.linspace(0, num_points - 1, num_total_points)
        
        # 선형 보간 수행
        way_x = np.interp(t_interpolated, t_original, x_vals)
        way_y = np.interp(t_interpolated, t_original, y_vals)
        way_z = np.interp(t_interpolated, t_original, z_vals)
        
        # 원래 경로점들의 보간된 점에서의 인덱스 계산
        original_indices_in_interpolated = np.searchsorted(t_interpolated, t_original)
        
        #############################Get Terrain of Way#########################
        #non interp
        int_way_x = list(map(int, way_x))
        int_way_y = list(map(int, way_y))
        terrain_z = np.array(z[int_way_x, int_way_y])
        
        ############################Add Waypoint###############################
        add_x_list=[]
        add_y_list=[]
        add_z_list=[]
        index_add=[]
        return_index_list=[]
        for i in range(len(waypoint_z)-1):
            return_x,return_y,return_z,return_index = self.add_waypoint(i,original_indices_in_interpolated,way_x,way_y,way_z,terrain_z)
            add_x_list.append(return_x)
            add_y_list.append(return_y)
            add_z_list.append(return_z)
            return_index_list.append(return_index)
            index_add.append(i+1)
        
        # 삽입을 위한 작업
        new_waypoint_x = waypoint_x.copy()
        new_waypoint_z = waypoint_z.copy()  # 원본 배열을 복사하여 작업
        new_waypoint_y = waypoint_y.copy() 
        new_index = original_indices_in_interpolated.copy()
        # 삽입 작업에 따른 위치 조정
        for i, (array, pos) in enumerate(zip(add_z_list, index_add)):
            # 배열이 삽입된 후에 뒤의 위치들은 삽입된 배열의 길이만큼 증가합니다.
            adjusted_pos = pos + sum(len(add_z_list[j]) for j in range(i))
            new_waypoint_z = np.insert(new_waypoint_z, adjusted_pos, array)
        
        for i, (array, pos) in enumerate(zip(add_y_list, index_add)):
            # 배열이 삽입된 후에 뒤의 위치들은 삽입된 배열의 길이만큼 증가합니다.
            adjusted_pos = pos + sum(len(add_y_list[j]) for j in range(i))
            new_waypoint_y = np.insert(new_waypoint_y, adjusted_pos, array)
        
        for i, (array, pos) in enumerate(zip(add_x_list, index_add)):
            # 배열이 삽입된 후에 뒤의 위치들은 삽입된 배열의 길이만큼 증가합니다.
            adjusted_pos = pos + sum(len(add_x_list[j]) for j in range(i))
            new_waypoint_x = np.insert(new_waypoint_x, adjusted_pos, array)
            
        for i, (array, pos) in enumerate(zip(return_index_list, index_add)):
            # 배열이 삽입된 후에 뒤의 위치들은 삽입된 배열의 길이만큼 증가합니다.
            adjusted_pos = pos + sum(len(return_index_list[j]) for j in range(i))
            new_index = np.insert(new_index, adjusted_pos, array)
        
        num_points = len(new_waypoint_x)
        num_total_points = 1000
        # x, y, z 좌표를 각각 분리
        x_vals = new_waypoint_x
        y_vals = new_waypoint_y
        z_vals = new_waypoint_z
        
        # 각 경로 구간에서 보간할 t 값 계산 (0에서 num_points-1까지)
        t_original = np.linspace(0, num_points - 1, num_points)
        t_interpolated = np.linspace(0, num_points - 1, num_total_points)
        
        # 선형 보간 수행
        way_x_2 = np.interp(t_interpolated, t_original, x_vals)
        way_y_2 = np.interp(t_interpolated, t_original, y_vals)
        way_z_2 = np.interp(t_interpolated, t_original, z_vals)
        
        # 원래 경로점들의 보간된 점에서의 인덱스 계산
        original_indices_in_interpolated_2 = np.searchsorted(t_interpolated, t_original)
        
        #non interp
        int_way_x_2 = list(map(int, way_x_2))
        int_way_y_2 = list(map(int, way_y_2))
        terrain_z_2 = np.array(z[int_way_x_2, int_way_y_2])
        
        add_x_list=[]
        add_y_list=[]
        add_z_list=[]
        index_add=[]
        return_index_list=[]
        for i in range(len(new_waypoint_z)-1):
            return_x,return_y,return_z,return_index = self.add_waypoint(i,original_indices_in_interpolated_2,way_x_2,way_y_2,way_z_2,terrain_z_2)
            add_x_list.append(return_x)
            add_y_list.append(return_y)
            add_z_list.append(return_z)
            return_index_list.append(return_index)
            index_add.append(i+1)
        
        # 삽입을 위한 작업
        new_waypoint_x_2 = new_waypoint_x.copy()
        new_waypoint_z_2 = new_waypoint_z.copy()  # 원본 배열을 복사하여 작업
        new_waypoint_y_2 = new_waypoint_y.copy() 
        new_index_2 = original_indices_in_interpolated_2.copy()
        # 삽입 작업에 따른 위치 조정
        for i, (array, pos) in enumerate(zip(add_z_list, index_add)):
            # 배열이 삽입된 후에 뒤의 위치들은 삽입된 배열의 길이만큼 증가합니다.
            adjusted_pos = pos + sum(len(add_z_list[j]) for j in range(i))
            new_waypoint_z_2 = np.insert(new_waypoint_z_2, adjusted_pos, array)
        
        for i, (array, pos) in enumerate(zip(add_y_list, index_add)):
            # 배열이 삽입된 후에 뒤의 위치들은 삽입된 배열의 길이만큼 증가합니다.
            adjusted_pos = pos + sum(len(add_y_list[j]) for j in range(i))
            new_waypoint_y_2 = np.insert(new_waypoint_y_2, adjusted_pos, array)
        
        for i, (array, pos) in enumerate(zip(add_x_list, index_add)):
            # 배열이 삽입된 후에 뒤의 위치들은 삽입된 배열의 길이만큼 증가합니다.
            adjusted_pos = pos + sum(len(add_x_list[j]) for j in range(i))
            new_waypoint_x_2 = np.insert(new_waypoint_x_2, adjusted_pos, array)
            
        for i, (array, pos) in enumerate(zip(return_index_list, index_add)):
            # 배열이 삽입된 후에 뒤의 위치들은 삽입된 배열의 길이만큼 증가합니다.
            adjusted_pos = pos + sum(len(return_index_list[j]) for j in range(i))
            new_index_2 = np.insert(new_index_2, adjusted_pos, array)

        return new_waypoint_x_2, new_waypoint_y_2,new_waypoint_z_2

    def find_shortest_path(self, nodes):
        graph = self.create_graph(nodes)
        try:
            # source와 target을 튜플로 변환
            source = tuple(nodes[0])
            target = tuple(nodes[-1])
            path = nx.dijkstra_path(graph, source=source, target=target)
        except nx.NetworkXNoPath:
            print("No path found. Returning direct path.")
            path = [nodes[0], nodes[-1]]
        except ValueError as e:
            print(f"Error in finding path: {e}. Returning direct path.")
            path = [nodes[0], nodes[-1]]
        return path



    def create_graph(self, nodes):
        graph = nx.Graph()
        elev_factor = 0.65
        dist_factor = 1 - elev_factor

        distances = []
        elevation_diffs = []

        for node in nodes:
            graph.add_node(tuple(node))

        for i, node1 in enumerate(nodes):
            for j, node2 in enumerate(nodes):
                if i != j:
                    distance = np.linalg.norm(np.array(node1) - np.array(node2))
                    if distance <= self.distance / 2:
                        elevation_diff = abs(
                            int(self.heightmap_resized[node1[0], node1[1]]) - int(self.heightmap_resized[node2[0], node2[1]]))
                        distances.append(distance)
                        elevation_diffs.append(elevation_diff)

        if distances and elevation_diffs:
            min_distance, max_distance = min(distances), max(distances)
            min_elevation_diff, max_elevation_diff = min(elevation_diffs), max(elevation_diffs)

            for i, node1 in enumerate(nodes):
                for j, node2 in enumerate(nodes):
                    if i != j:
                        distance = np.linalg.norm(np.array(node1) - np.array(node2))
                        if distance <= self.distance / 2:
                            elevation_diff = abs(
                                int(self.heightmap_resized[node1[0], node1[1]]) - int(self.heightmap_resized[node2[0], node2[1]]))

                            normalized_distance = (distance - min_distance) / (
                                    max_distance - min_distance) if max_distance != min_distance else 0
                            normalized_elevation_diff = (elevation_diff - min_elevation_diff) / (
                                    max_elevation_diff - min_elevation_diff) if max_elevation_diff != min_elevation_diff else 0

                            weight = dist_factor * normalized_distance + elev_factor * normalized_elevation_diff
                            weight = max(weight, 1e-6)  # 가중치가 0이 되지 않도록 함
                            graph.add_edge(tuple(node1), tuple(node2), weight=weight)
        return graph
    
    


    def plot_path_2d(self, output_path):
        plt.figure(figsize=(10, 10))
        plt.imshow(self.heightmap, cmap='gray')
        plt.plot(self.path_y, self.path_x, 'r-')
        plt.plot(self.path_y[0], self.path_x[0], 'go', markersize=10, label='Start')
        plt.plot(self.path_y[-1], self.path_x[-1], 'bo', markersize=10, label='Goal')
        plt.legend()
        plt.title('2D Path on Heightmap')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.savefig(output_path)
        plt.close()

    def plot_path_3d(self, output_path):
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')

        # Plot the heightmap as a surface
        x = np.arange(0, self.heightmap.shape[1], 1)
        y = np.arange(0, self.heightmap.shape[0], 1)
        X, Y = np.meshgrid(x, y)
        ax.plot_surface(X, Y, self.heightmap, cmap='terrain', alpha=0.5)

        # Plot the path
        ax.plot(self.path_y, self.path_x, self.path_z, 'r-', linewidth=2)
        ax.scatter(self.path_y[0], self.path_x[0], self.path_z[0], c='g', s=100, label='Start')
        ax.scatter(self.path_y[-1], self.path_x[-1], self.path_z[-1], c='b', s=100, label='Goal')

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.legend()
        ax.set_title('3D Path on Heightmap')
        plt.savefig(output_path)
        plt.close()
    
    def plot_path_2d_learning(self, output_path):
        plt.figure(figsize=(10, 10))
        plt.imshow(self.heightmap_resized, cmap='gray')
        plt.plot(self.path_y_learning, self.path_x_learning, 'r-')
        plt.plot(self.path_y_learning[0], self.path_x_learning[0], 'go', markersize=10, label='Start')
        plt.plot(self.path_y_learning[-1], self.path_x_learning[-1], 'bo', markersize=10, label='Goal')
        plt.legend()
        plt.title('2D Path on Heightmap of learning model')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.savefig(output_path)
        plt.close()

    def plot_path_3d_learning(self, output_path):
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')

        # Plot the heightmap as a surface
        x = np.arange(0, self.heightmap_resized.shape[1], 1)
        y = np.arange(0, self.heightmap_resized.shape[0], 1)
        X, Y = np.meshgrid(x, y)
        ax.plot_surface(X, Y, self.heightmap_resized, cmap='terrain', alpha=0.5)

        # Plot the path
        ax.plot(self.path_y_learning, self.path_x_learning, self.path_z_learning, 'r-', linewidth=2)
        ax.scatter(self.path_y_learning[0], self.path_x_learning[0], self.path_z_learning[0], c='g', s=100, label='Start')
        ax.scatter(self.path_y_learning[-1], self.path_x_learning[-1], self.path_z_learning[-1], c='b', s=100, label='Goal')

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.legend()
        ax.set_title('3D Path on Heightmap of learning model')
        plt.savefig(output_path)
        plt.close()





    def plot_binary(self, output_path):
        # Implementation of plot_binary method
        pass

    def plot_original(self, output_path):
        # Implementation of plot_original method
        pass

    def print_distance_length(self):
        total_wp_distance = self.total_waypoint_distance()
        init_target_distance = self.init_to_target_distance()

        length = total_wp_distance
        print("Path Length: {:.2f}".format(length))

        return length

    def total_waypoint_distance(self):
        total_distance = 0
        for i in range(1, len(self.path_x)):
            dx = self.path_x[i] - self.path_x[i - 1]
            dy = self.path_y[i] - self.path_y[i - 1]
            total_distance += np.sqrt(dx ** 2 + dy ** 2)
        return total_distance

    def init_to_target_distance(self):
        dx = self.path_x[-1] - self.path_x[0]
        dy = self.path_y[-1] - self.path_y[0]
        return np.sqrt(dx ** 2 + dy ** 2)


#############################################################################################################
# added by controller
# altitude 150 -> 5

#        self.path_z = 5 * np.ones(len(self.path_x))


#############################################################################################################


class RRT:
    def __init__(self, model_path, image_path, map_size=1000):
        self.model = onnx.load(model_path)
        self.ort_session = ort.InferenceSession(model_path)
        self.map_size = map_size
        self.raw_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        self.raw_image_flipped = cv2.flip(self.raw_image, 0)
        self.image_new = np.where(self.raw_image_flipped < 150, 0, 1)  # 130
        # Heightmap 하얀 부분이 더 높은 장애물임
        # 150m로 이동할 때, 장애물이 150보다 작으면 지나갈 수 있으니 0 150보다 크면 1
        # 150m 이상 높은 장애물 모두 통과 가능하도록 경로 산출

    # Definition
    def collision_check(self, Map, from_wp, to_wp):
        N_grid = len(Map) + 5000

        min_x = math.floor(min(np.round(from_wp[0]), np.round(to_wp[0])))
        max_x = math.ceil(max(np.round(from_wp[0]), np.round(to_wp[0])))
        min_y = math.floor(min(np.round(from_wp[1]), np.round(to_wp[1])))
        max_y = math.ceil(max(np.round(from_wp[1]), np.round(to_wp[1])))

        if max_x > N_grid - 1:
            max_x = N_grid - 1
        if max_y > N_grid - 1:
            max_y = N_grid - 1

        check1 = Map[min_y][min_x]
        check2 = Map[min_y][max_x]
        check3 = Map[max_y][min_x]
        check4 = Map[max_y][max_x]

        flag_collision = max(check1, check2, check3, check4)

        return flag_collision

    def RRT_PathPlanning(self, Start, Goal):

        TimeStart = time.time()

        # Initialization
        Image = self.image_new

        # N_grid = len(Image)
        N_grid = 5000

        # print(Start)
        Init = np.array([Start[0], 2, Start[1]])
        Target = np.array([Goal[0], 2, Goal[1]])

        Start = np.array([[Init[0]], [Init[2]]])
        Goal = np.array([[Target[0]], [Target[2]]])

        Start = Start.astype(float)
        Goal = Goal.astype(float)

        # User Parameter
        step_size = np.linalg.norm(Start - Goal, 2) / 500
        Search_Margin = 0

        ##.. Algorithm Initialize
        q_start = np.array([Start, 0, 0], dtype=object)  # Coord, Cost, Parent
        q_goal = np.array([Goal, 0, 0], dtype=object)

        idx_nodes = 1

        nodes = q_start
        nodes = np.vstack([nodes, q_start])
        # np.vstack([q_start, q_goal])
        ##.. Algorithm Start

        flag_end = 0
        N_Iter = 0
        while (flag_end == 0):
            # Set Searghing Area
            Search_Area_min = Goal - Search_Margin
            Search_Area_max = Goal + Search_Margin
            q_rand = Search_Area_min + (Search_Area_max - Search_Area_min) * np.random.uniform(0, 1, [2, 1])

            # Pick the closest node from existing list to branch out from
            dist_list = []
            for i in range(0, idx_nodes + 1):
                dist = np.linalg.norm(nodes[i][0] - q_rand)
                if (i == 0):
                    dist_list = [dist]
                else:
                    dist_list.append(dist)

            val = min(dist_list)
            idx = dist_list.index(val)

            q_near = nodes[idx]
            # q_new = Tree()
            # q_new = collections.namedtuple('Tree', ['coord', 'cost', 'parent'])
            new_coord = q_near[0] + (q_rand - q_near[0]) / val * step_size

            # Collision Check
            flag_collision = self.collision_check(Image, q_near[0], new_coord)
            # print(q_near[0], new_coord)

            # flag_collision = 0

            # Add to Tree
            if (flag_collision == 0):
                Search_Margin = 0
                new_cost = nodes[idx][1] + np.linalg.norm(new_coord - q_near[0])
                new_parent = idx
                q_new = np.array([new_coord, new_cost, new_parent], dtype=object)
                # print(nodes[0])

                nodes = np.vstack([nodes, q_new])
                # nodes = list(zip(nodes, q_new))
                # nodes.append(q_new)
                # print(nodes[0])

                Goal_Dist = np.linalg.norm(new_coord - q_goal[0])

                idx_nodes = idx_nodes + 1

                if (Goal_Dist < step_size):
                    flag_end = 1
                    nodes = np.vstack([nodes, q_goal])
                    idx_nodes = idx_nodes + 1
            else:
                Search_Margin = Search_Margin + N_grid / 100

                if Search_Margin >= N_grid:
                    Search_Margin = N_grid - 1
            N_Iter = N_Iter + 1
            if N_Iter > 100000:
                break

        flag_merge = 0
        idx = 0
        idx_parent = idx_nodes - 1
        path_x_inv = np.array([])
        path_y_inv = np.array([])
        while (flag_merge == 0):
            path_x_inv = np.append(path_x_inv, nodes[idx_parent][0][0])
            path_y_inv = np.append(path_y_inv, nodes[idx_parent][0][1])

            idx_parent = nodes[idx_parent][2]
            idx = idx + 1

            if idx_parent == 0:
                flag_merge = 1

        path_x = np.array([])
        path_y = np.array([])
        for i in range(0, idx - 2):
            path_x = np.append(path_x, path_x_inv[idx - i - 1])
            path_y = np.append(path_y, path_y_inv[idx - i - 1])

        self.path_x = path_x
        self.path_y = path_y
        self.path_z = 150 * np.ones(len(self.path_x))

        TimeEnd = time.time()

    def plot_RRT(self, output_path):

        MapSize = self.map_size

        ## Plot and Save Image
        path_x = self.path_x
        path_y = self.path_y

        ## Plot and Save Image
        imageLine2 = self.raw_image

        # 이미지에 맞게 SAC Waypoint 변경 후 그리기
        for m in range(0, len(path_x) - 2):
            Im_i = int(path_x[m + 1])
            Im_j = MapSize - int(path_y[m + 1])

            Im_iN = int(path_x[m + 2])
            Im_jN = MapSize - int(path_y[m + 2])

            # 각 웨이포인트에 점 찍기 (thickness 2)
            cv2.circle(imageLine2, (Im_i, Im_j), radius=2, color=(0, 255, 0), thickness=1)

            # 웨이포인트 사이를 선으로 연결 (thickness 1)
            cv2.line(imageLine2, (Im_i, Im_j), (Im_iN, Im_jN), (0, 255, 0), thickness=1, lineType=cv2.LINE_AA)

        cv2.imwrite(output_path, imageLine2)  ################################

    def plot_RRT_binary(self, output_path):
        MapSize = self.map_size

        ## Plot and Save Image
        path_x = self.path_x
        path_y = self.path_y

        Image_New = self.image_new
        Image_New2 = Image_New * 255
        Image_New2 = np.uint8(np.uint8((255 - Image_New2)))

        # Image_New2 = cv2.flip(Image_New2, 0)
        Image_New2 = cv2.flip(Image_New2, 1)
        Image_New2 = cv2.rotate(Image_New2, cv2.ROTATE_90_CLOCKWISE)
        Image_New2 = cv2.rotate(Image_New2, cv2.ROTATE_90_CLOCKWISE)
        # Image_New2 = cv2.rotate(Image_New2, cv2.ROTATE_90_CLOCKWISE)
        # Image_New2 = cv2.rotate(Image_New2, cv2.ROTATE_90_CLOCKWISE)
        imageLine = Image_New2.copy()
        # 이미지 크기에 따른 그리드 간격 설정
        grid_interval = 20

        # Image_New2 이미지에 그리드 그리기
        for x in range(0, imageLine.shape[1], grid_interval):  # 이미지의 너비에 따라
            cv2.line(imageLine, (x, 0), (x, imageLine.shape[0]), color=(125, 125, 125), thickness=2)

        for y in range(0, imageLine.shape[0], grid_interval):  # 이미지의 높이에 따라
            cv2.line(imageLine, (0, y), (imageLine.shape[1], y), color=(125, 125, 125), thickness=1)

        # 이미지에 맞게 SAC Waypoint 변경 후 그리기
        for i in range(1, len(path_x) - 2):  # Changed to step_num - 1
            for m in range(0, len(path_x) - 2):
                Im_i = int(path_x[m + 1])
                Im_j = MapSize - int(path_y[m + 1])

                Im_iN = int(path_x[m + 2])
                Im_jN = MapSize - int(path_y[m + 2])

                # 각 웨이포인트에 점 찍기 (thickness 2)
                cv2.circle(imageLine, (Im_i, Im_j), radius=2, color=(0, 255, 0), thickness=1)

                # 웨이포인트 사이를 선으로 연결 (thickness 1)
                cv2.line(imageLine, (Im_i, Im_j), (Im_iN, Im_jN), (0, 255, 0), thickness=1, lineType=cv2.LINE_AA)

        cv2.imwrite(output_path, imageLine)  ################################

    def calculate_and_print_path_info(self):
        LenRRT = 0
        for cal in range(len(self.path_x) - 1):
            First = np.array([self.path_x[cal], self.path_y[cal]])
            Second = np.array([self.path_x[cal + 1], self.path_y[cal + 1]])

            U = (Second - First) / np.linalg.norm(Second - First)

            State = First
            for cal_step in range(500):
                State = State + U

                if np.linalg.norm(Second - State) < 20:
                    break

                # Add collision check code here if needed

            Len_temp = np.linalg.norm(Second - First)
            LenRRT += Len_temp

        print("RRT 경로 길이:", LenRRT)
        return LenRRT

    def total_waypoint_distance(self):
        total_distance = 0
        for i in range(1, len(self.path_x)):
            dx = self.path_x[i] - self.path_x[i - 1]
            dy = self.path_y[i] - self.path_y[i - 1]
            total_distance += np.sqrt(dx ** 2 + dy ** 2)
        return total_distance

    def init_to_target_distance(self):
        dx = self.path_x[-1] - self.path_x[0]
        dy = self.path_y[-1] - self.path_y[0]
        return np.sqrt(dx ** 2 + dy ** 2)

    def print_distance_length(self):
        total_wp_distance = self.total_waypoint_distance()
        init_target_distance = self.init_to_target_distance()

        # # 절대오차 계산
        # absolute_error = abs(total_wp_distance - init_target_distance)
        #
        # # 상대오차 계산 (0으로 나누는 경우 예외 처리)
        # if init_target_distance != 0:
        #     relative_error = absolute_error / init_target_distance
        #     print(f"절대오차: {absolute_error:.2f}, 상대오차: {relative_error:.2%}")
        # else:
        #     print(f"절대오차: {absolute_error:.2f}, 상대오차: 계산 불가 (분모가 0)")

        length = total_wp_distance
        print("RRT: Path Length: {:.2f}".format(length))

        return length


class PathPlanningServer(Node):  # topic 이름과 message 타입은 서로 매칭되어야 함

    def __init__(self):
        super().__init__('minimal_subscriber')

        #self.bridge = CvBridge()

        # mode change
        self.mode = 1

        # initialize global waypoint
        self.Init_custom = [0.0, 0.0, 0.0]
        self.Target_custom = [0.0, 0.0, 0.0]

        # Initialiaztion
        ## Range [-2500, 2500]으로 바꾸기
        self.MapSize = 1000  # size 500
        self.Step_Num_custom = self.MapSize + 1000

        #############################################################################################################
        # added by controller
        # file path
        self.image_path = '/home/user/workspace/ros2/ros2_ws/src/pathplanning/pathplanning/map/512-001.png'
        self.model_path = "/home/user/workspace/ros2/ros2_ws/src/pathplanning/pathplanning/model/sac_model_85000.onnx"

        # path plannig complete flag
        self.path_plannig_start = False  # flag whether path planning start
        self.path_planning_complete = False  # flag whether path planning is complete

        # heartbeat signal of another module node
        self.controller_heartbeat = False
        self.path_following_heartbeat = False
        self.collision_avoidance_heartbeat = False

        # declare global waypoint subscriber from controller
        self.global_waypoint_subscriber = self.create_subscription(GlobalWaypointSetpoint, '/global_waypoint_setpoint',
                                                                   self.global_waypoint_callback, 10)

        # declare heartbeat_subscriber
        self.controller_heartbeat_subscriber = self.create_subscription(Heartbeat, '/controller_heartbeat',
                                                                        self.controller_heartbeat_call_back, 10)
        self.path_following_heartbeat_subscriber = self.create_subscription(Heartbeat, '/path_following_heartbeat',
                                                                            self.path_following_heartbeat_call_back, 10)
        self.collision_avoidance_heartbeat_subscriber = self.create_subscription(Heartbeat,
                                                                                 '/collision_avoidance_heartbeat',
                                                                                 self.collision_avoidance_heartbeat_call_back,
                                                                                 10)

        # declare local waypoint publisher to controller
        self.local_waypoint_publisher = self.create_publisher(LocalWaypointSetpoint, '/local_waypoint_setpoint_from_PP',
                                                              10)

        # declare heartbeat_publisher
        self.heartbeat_publisher = self.create_publisher(Heartbeat, '/path_planning_heartbeat', 10)

        print("                                          ")
        print("===== Path Planning Node is Running  =====")
        print("                                          ")

        # declare heartbeat_timer
        period_heartbeat_mode = 1
        self.heartbeat_timer = self.create_timer(period_heartbeat_mode, self.publish_heartbeat)

    #############################################################################################################

    #############################################################################################################
    # added by controller

    # publish local waypoint and path planning complete flag
    def local_waypoint_publish(self):
        msg = LocalWaypointSetpoint()
        msg.path_planning_complete = self.path_planning_complete
        msg.waypoint_x = self.waypoint_x
        msg.waypoint_y = self.waypoint_y
        # qgc coordinate (y, x ,z )
        msg.waypoint_z = self.waypoint_z
        self.local_waypoint_publisher.publish(msg)
        print("                                          ")
        print("==  Sended local waypoint to controller ==")
        print("                                          ")

    # heartbeat check function
    # heartbeat publish
    def publish_heartbeat(self):
        msg = Heartbeat()
        msg.heartbeat = True
        self.heartbeat_publisher.publish(msg)

    # heartbeat subscribe from controller
    def controller_heartbeat_call_back(self, msg):
        self.controller_heartbeat = msg.heartbeat

    # heartbeat subscribe from path following
    def path_following_heartbeat_call_back(self, msg):
        self.path_following_heartbeat = msg.heartbeat

    # heartbeat subscribe from collision avoidance
    def collision_avoidance_heartbeat_call_back(self, msg):
        self.collision_avoidance_heartbeat = msg.heartbeat

    #############################################################################################################

    # added by controller
    # update global waypoint and path plannig start flag if subscribe global waypoint from controller
    def global_waypoint_callback(self, msg):
        # check heartbeat
        if self.controller_heartbeat and self.path_following_heartbeat and self.collision_avoidance_heartbeat:
            print("i am here 1")
            if not self.path_plannig_start and not self.path_planning_complete:
                print("i am here 2")  
                self.Init_custom = msg.start_point
                self.Target_custom = msg.goal_point
                self.path_plannig_start = True

                print("                                          ")
                print("===== Received Path Planning Request =====")
                print("                                          ")

                if self.mode == 1 and not self.path_planning_complete:
                    # start path planning
                    planner = PathPlanning(self.model_path, self.image_path, self.Init_custom, self.Target_custom)
                    planner.plan_path(self.Init_custom, self.Target_custom)

                    #planner.plot_binary(
                    #    "/home/user/workspace/ros2/ros2_ws/src/pathplanning/pathplanning/Results_Images/SAC_Result_biary.png")
                    #planner.plot_original(
                    #    "/home/user/workspace/ros2/ros2_ws/src/pathplanning/pathplanning/Results_Images/SAC_Result_og.png")
                    print("                                          ")
                    print("=====   Path Planning Complete!!     =====")
                    print("                                          ")

                    planner.print_distance_length()
                    print("                                           ")

                    # setting msg
                    self.path_planning_complete = True
                    self.waypoint_x = planner.path_x.tolist()
                    self.waypoint_y = planner.path_y.tolist()
                    self.waypoint_z = planner.path_z.tolist()

                    print('+++++++++++++++++++++++++++++')
                    print(self.waypoint_x)
                    print(self.waypoint_y)
                    print(self.waypoint_z)

                    # publish local waypoint and path planning complete flag
                    self.local_waypoint_publish()

                elif self.mode == 2:
                    # Implement mode 2 logic here if needed
                    pass

                elif self.mode == 3:
                    # Implement mode 3 logic here if needed
                    pass
        else:
            pass


def main(args=None):
    rclpy.init(args=args)
    SAC_module = PathPlanningServer()
    try:
        rclpy.spin(SAC_module)
    except KeyboardInterrupt:
        SAC_module.get_logger().info('Keyboard Interrupt (SIGINT)')
    finally:
        SAC_module.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
