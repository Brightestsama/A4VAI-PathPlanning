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
from .Plan2WP_core import PathPlannerCore


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

        self.z_offset = 200
        self.height_scale = 1000/65535

        #############################################################################################################
        # added by controller
        # file path
        self.image_path = "/home/user/workspace/ros2/ros2_ws/src/pathplanning/pathplanning/map/expanded-1000.png"

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
                    planner = PathPlannerCore(
                        self.image_path,
                        self.Init_custom,
                        self.Target_custom,
                        self.z_offset,
                        self.height_scale
                    )
                    
                    waypoint_x, waypoint_y, waypoint_z = planner.plan()

                    print("                                          ")
                    print("=====   Path Planning Complete!!     =====")
                    print("                                          ")


                    # setting msg
                    self.path_planning_complete = True
                    
                    self.waypoint_x = [float(x) for x in waypoint_x]
                    self.waypoint_y = [float(x) for x in waypoint_y]
                    self.waypoint_z = [float(x) for x in waypoint_z]

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
    Astar_module = PathPlanningServer()
    try:
        rclpy.spin(Astar_module)
    except KeyboardInterrupt:
        Astar_module.get_logger().info("Keyboard Interrupt (SIGINT)")
    finally:
        Astar_module.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
