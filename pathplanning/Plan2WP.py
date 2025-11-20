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
from std_msgs.msg import Bool
import numpy as np
import cv2
from gymnasium import spaces
import torch.nn.functional as F
import networkx as nx
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os
import time
import tracemalloc
import tensorrt as trt
import pycuda.autoinit
import pycuda.driver as cuda

#############################################################################################################
# added by controller
from custom_msgs.msg import GlobalWaypointSetpoint, LocalWaypointSetpoint


#############################################################################################################
class PathPlanning:
    def __init__(
        self,
        model_path,
        heightmap_path,
        start,
        goal,
        n_waypoints=6,
        scale_factor=60,
        image_size=60,
        z_factor=3,
    ):

        self.trt_engine_path = model_path
        self.heightmap_path = heightmap_path

        self.start_z = start[2]
        self.goal_z = goal[2]
        self.image_size = image_size

        self.n_waypoints = n_waypoints
        self.scale_factor = scale_factor
        self.z_factor = z_factor  # New z_factor attribute
        # Load and preprocess heightmap
        self.heightmap = self.load_heightmap(heightmap_path)
        self.h, self.w = self.heightmap.shape

        original_heightmap = cv2.imread(heightmap_path, cv2.IMREAD_GRAYSCALE)
        self.original_heightmap = cv2.normalize(
            original_heightmap, None, 0, 255, cv2.NORM_MINMAX
        ).astype(np.uint8)
        self.scale_factor_waypoint_x = (
            self.original_heightmap.shape[1] / self.heightmap.shape[1]
        )  # Scale Factor of waypoint
        self.scale_factor_waypoint_y = (
            self.original_heightmap.shape[0] / self.heightmap.shape[0]
        )  # Scale Factor of waypoint
        self.start = [
            start[1] / self.scale_factor_waypoint_x,
            start[0] / self.scale_factor_waypoint_y,
        ]
        self.goal = [
            goal[1] / self.scale_factor_waypoint_x,
            goal[0] / self.scale_factor_waypoint_y,
        ]

        self.min_distance_ratio = 0.3
        self.square_size = min(self.h, self.w)
        self.min_distance = int(self.square_size * self.min_distance_ratio)
        # Check distance between start and goal
        if np.linalg.norm(np.array(start) - np.array(goal)) < self.min_distance:
            raise ValueError("Start and Goal is too close")

    def load_heightmap(self, path):
        # Load the image
        image = cv2.imread(path, cv2.IMREAD_UNCHANGED)

        if image is None:
            raise ValueError(f"Failed to load heightmap from path: {path}")

        # Check if the image is already grayscale
        if len(image.shape) == 3:
            # Convert to grayscale if it's not
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Ensure the image is square
        height, width = image.shape
        size = min(height, width)
        image = image[:size, :size]

        # Resize to nearest 2^(n-1) + 1
        # target_size = 1024
        target_size = 2 ** (int(np.log2(size - 1))) + 1
        if size != target_size:
            image = cv2.resize(
                image, (target_size, target_size), interpolation=cv2.INTER_AREA
            )

        return image  # , rotation

    def plan_path(self):
        # Memory Usage Check
        tracemalloc.start()

        # Start Processing Time Check
        start_times = os.times()
        wall_clock_start = time.time()

        # Tensor RT Path planning
        tensorrt_infer = self.load_tensorrt_engine(self.trt_engine_path)

        # environment reset
        trt_obs, info = self.reset()  # reset은 이미 설정된 start, goal을 사용
        trt_obs = self._get_obs()

        done = False
        while not done:
            trt_obs = trt_obs.astype(np.float32)
            trt_obs = np.expand_dims(trt_obs, axis=0)
            trt_action = tensorrt_infer(trt_obs)[1]
            # 환경 스텝 진행
            trt_obs, done, _ = self.step(trt_action)

        # END Processing Time Check
        end_times = os.times()
        wall_clock_end = time.time()
        user_time = end_times.user - start_times.user
        system_time = end_times.system - start_times.system
        elapsed_wall_clock = wall_clock_end - wall_clock_start

        print("TensorRT Processing User CPU Time [sec] :", user_time)
        print("TensorRT Processing System CPU Time [sec] :", system_time)
        print("TensorRT Processing Wall-Clock Time [sec] :", elapsed_wall_clock)

        # 최종 결과 저장
        trt_path = self.current_agent1_path
        final_trt_reward, trt_path_ratio = self.calculate_3d_path_reward2_og_og(
            self.current_agent1_path
        )
        print("TensorRT Path Ratio :", trt_path_ratio)

        self.path_x_learning = [p[1] for p in trt_path]
        self.path_y_learning = [p[0] for p in trt_path]
        self.path_z_learning = [
            self.heightmap[int(p[0]), int(p[1])] + self.z_factor for p in trt_path
        ]

        self.scaled_path_x = [p[1] * self.scale_factor_waypoint_x for p in trt_path]
        self.scaled_path_y = [p[0] * self.scale_factor_waypoint_y for p in trt_path]
        self.scaled_path_z = [
            self.original_heightmap[
                int(p[0] * self.scale_factor_waypoint_y),
                int(p[1] * self.scale_factor_waypoint_x),
            ]
            * 0.1
            for p in trt_path
        ]
        self.scaled_path_z[0] = self.start_z
        self.scaled_path_z[-1] = self.goal_z

        self.path_y, self.path_x, self.path_z = self.add_waypoint_main_2(
            self.scaled_path_y,
            self.scaled_path_x,
            self.scaled_path_z,
            self.original_heightmap * 0.1,
        )
        self.path_z = self.path_z + self.z_factor
        path_final_3D_learning_model = np.column_stack(
            (self.path_x_learning, self.path_y_learning, self.path_z_learning)
        )  # output path of learning model scaled target size
        path_final_3D = np.column_stack(
            (self.path_x, self.path_y, self.path_z)
        )  # real path

        final_path_ratio = self.calculate_real_path_ratio(
            self.path_x, self.path_y, self.path_z
        )
        print("Final Path Ratio :", final_path_ratio)
        print("Output path of learning model :", path_final_3D_learning_model)
        print("Output Real Path", path_final_3D)

        # Memory Check END
        current, peak = tracemalloc.get_traced_memory()
        print(f"Current Memory Usage : {current / (1024*1024):.2f} MB")
        print(f"Peak Memory Usage : {peak / (1024*1024):.2f} MB")

        # Check if /home/user/workspace/ros2/ros2_ws/src/pathplanning/pathplanning/Results_Images exists
        if not os.path.exists(
            "/home/user/workspace/ros2/ros2_ws/src/pathplanning/pathplanning/Results_Images"
        ):
            os.makedirs(
                "/home/user/workspace/ros2/ros2_ws/src/pathplanning/pathplanning/Results_Images"
            )
        else:
            print("Results_Images directory already exists")
            # remove all files in Results_Images
            print("Removing all files in Results_Images")
            for file in os.listdir(
                "/home/user/workspace/ros2/ros2_ws/src/pathplanning/pathplanning/Results_Images"
            ):
                os.remove(
                    os.path.join(
                        "/home/user/workspace/ros2/ros2_ws/src/pathplanning/pathplanning/Results_Images",
                        file,
                    )
                )

        # 경로생성 결과 확인용
        self.plot_path_2d(
            "/home/user/workspace/ros2/ros2_ws/src/pathplanning/pathplanning/Results_Images/path_2d.png"
        )
        self.plot_path_3d(
            "/home/user/workspace/ros2/ros2_ws/src/pathplanning/pathplanning/Results_Images/path_3d.png"
        )
        self.plot_path_2d_learning(
            "/home/user/workspace/ros2/ros2_ws/src/pathplanning/pathplanning/Results_Images/path_2d_learning.png"
        )
        self.plot_path_3d_learning(
            "/home/user/workspace/ros2/ros2_ws/src/pathplanning/pathplanning/Results_Images/path_3d_learning.png"
        )

    def plot_path_2d(self, output_path):
        plt.figure(figsize=(10, 10))
        plt.imshow(self.original_heightmap, cmap="gray")
        plt.plot(self.path_x, self.path_y, "r-")
        plt.plot(self.path_x[0], self.path_y[0], "go", markersize=10, label="Start")
        plt.plot(self.path_x[-1], self.path_y[-1], "bo", markersize=10, label="Goal")
        plt.legend()
        plt.title("2D Path on Heightmap")
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.savefig(output_path)
        plt.close()

    def plot_path_3d(self, output_path):
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection="3d")

        # Plot the heightmap as a surface
        x = np.arange(0, self.original_heightmap.shape[1], 1)
        y = np.arange(0, self.original_heightmap.shape[0], 1)
        X, Y = np.meshgrid(x, y)
        ax.plot_surface(X, Y, self.original_heightmap * 0.1, cmap="terrain", alpha=0.5)

        # Plot the path
        ax.plot(self.path_x, self.path_y, self.path_z, "r-", linewidth=2)
        ax.scatter(
            self.path_x[0], self.path_y[0], self.path_z[0], c="g", s=100, label="Start"
        )
        ax.scatter(
            self.path_x[-1],
            self.path_y[-1],
            self.path_z[-1],
            c="b",
            s=100,
            label="Goal",
        )

        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.legend()
        ax.set_title("3D Path on Heightmap")
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
            total_distance += np.sqrt(dx**2 + dy**2)
        return total_distance

    def init_to_target_distance(self):
        dx = self.path_x[-1] - self.path_x[0]
        dy = self.path_y[-1] - self.path_y[0]
        return np.sqrt(dx**2 + dy**2)

class PathPlanningServer(Node):  # topic 이름과 message 타입은 서로 매칭되어야 함

    def __init__(self):
        super().__init__("minimal_subscriber")

        # self.bridge = CvBridge()

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
        self.image_path = "/home/user/workspace/ros2/ros2_ws/src/pathplanning/pathplanning/map/512-001.png"
        self.model_path = "/home/user/workspace/ros2/ros2_ws/src/pathplanning/pathplanning/model/weight.onnx_fp16.trt"

        # path plannig complete flag
        self.path_plannig_start = False  # flag whether path planning start
        self.path_planning_complete = False  # flag whether path planning is complete

        # heartbeat signal of another module node
        self.controller_heartbeat = False
        self.path_following_heartbeat = False
        self.collision_avoidance_heartbeat = False

        # declare global waypoint subscriber from controller
        self.global_waypoint_subscriber = self.create_subscription(
            GlobalWaypointSetpoint,
            "/global_waypoint_setpoint",
            self.global_waypoint_callback,
            10,
        )

        # declare heartbeat_subscriber
        self.controller_heartbeat_subscriber = self.create_subscription(
            Bool, "/controller_heartbeat", self.controller_heartbeat_call_back, 10
        )
        self.path_following_heartbeat_subscriber = self.create_subscription(
            Bool,
            "/path_following_heartbeat",
            self.path_following_heartbeat_call_back,
            10,
        )
        self.collision_avoidance_heartbeat_subscriber = self.create_subscription(
            Bool,
            "/collision_avoidance_heartbeat",
            self.collision_avoidance_heartbeat_call_back,
            10,
        )

        # declare local waypoint publisher to controller
        self.local_waypoint_publisher = self.create_publisher(
            LocalWaypointSetpoint, "/local_waypoint_setpoint_from_PP", 10
        )

        # declare heartbeat_publisher
        self.heartbeat_publisher = self.create_publisher(
            Bool, "/path_planning_heartbeat", 10
        )

        print("                                          ")
        print("===== Path Planning Node is Running  =====")
        print("                                          ")

        # declare heartbeat_timer
        period_heartbeat_mode = 1
        self.heartbeat_timer = self.create_timer(
            period_heartbeat_mode, self.publish_heartbeat
        )

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
        msg = Bool()
        msg.data = True
        self.heartbeat_publisher.publish(msg)

    # heartbeat subscribe from controller
    def controller_heartbeat_call_back(self, msg):
        self.controller_heartbeat = msg.data

    # heartbeat subscribe from path following
    def path_following_heartbeat_call_back(self, msg):
        self.path_following_heartbeat = msg.data

    # heartbeat subscribe from collision avoidance
    def collision_avoidance_heartbeat_call_back(self, msg):
        self.collision_avoidance_heartbeat = msg.data

    #############################################################################################################

    # added by controller
    # update global waypoint and path plannig start flag if subscribe global waypoint from controller
    def global_waypoint_callback(self, msg):
        # check heartbeat
        if (
            self.controller_heartbeat
            and self.path_following_heartbeat
            and self.collision_avoidance_heartbeat
        ):
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
                    planner = PathPlanning(
                        self.model_path,
                        self.image_path,
                        self.Init_custom,
                        self.Target_custom,
                    )
                    planner.plan_path()

                    # planner.plot_binary(
                    #    "/home/user/workspace/ros2/ros2_ws/src/pathplanning/pathplanning/Results_Images/SAC_Result_biary.png")
                    # planner.plot_original(
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

                    print("+++++++++++++++++++++++++++++")
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
        SAC_module.get_logger().info("Keyboard Interrupt (SIGINT)")
    finally:
        SAC_module.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
