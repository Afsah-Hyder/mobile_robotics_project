import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseWithCovarianceStamped
import numpy as np
import matplotlib.pyplot as plt


class PoseDifferenceNode(Node):
    def __init__(self):
        super().__init__('pose_difference_node')

        # Subscriptions
        self.wheel_odom_sub = self.create_subscription(
            Odometry,
            '/wheel/odometry',
            self.wheel_odom_callback,
            10
        )

        self.amcl_pose_sub = self.create_subscription(
            PoseWithCovarianceStamped,
            '/amcl_pose',
            self.amcl_pose_callback,
            10
        )

        # Internal storage for poses
        self.wheel_pose = None
        self.amcl_pose = None

        # Lists to store individual errors for plotting
        self.x_errors = []
        self.y_errors = []
        self.theta_errors = []

    def wheel_odom_callback(self, msg):
        self.wheel_pose = msg.pose.pose
        self.calculate_errors()

    def amcl_pose_callback(self, msg):
        self.amcl_pose = msg.pose.pose
        self.calculate_errors()

    def calculate_errors(self):
        if self.wheel_pose is None or self.amcl_pose is None:
            return

        # Extract position and orientation
        wheel_position = np.array([self.wheel_pose.position.x, self.wheel_pose.position.y])
        amcl_position = np.array([self.amcl_pose.position.x, self.amcl_pose.position.y])

        # Calculate position errors (x, y)
        x_error = wheel_position[0] - amcl_position[0]
        y_error = wheel_position[1] - amcl_position[1]

        # Calculate orientation error (theta) using quaternion difference
        wheel_orientation = np.array([self.wheel_pose.orientation.x, 
                                      self.wheel_pose.orientation.y,
                                      self.wheel_pose.orientation.z, 
                                      self.wheel_pose.orientation.w])
        amcl_orientation = np.array([self.amcl_pose.orientation.x, 
                                     self.amcl_pose.orientation.y,
                                     self.amcl_pose.orientation.z, 
                                     self.amcl_pose.orientation.w])
        
        # Calculate the angle difference between the two orientations
        angle_diff = 2 * np.arccos(np.abs(np.dot(wheel_orientation, amcl_orientation)))

        # Store errors in their respective lists
        self.x_errors.append(x_error)
        self.y_errors.append(y_error)
        self.theta_errors.append(angle_diff)
        self.get_logger().info(f"x error: {x_error}, y error: {y_error}, theta error: {angle_diff}")

    def plot_errors(self):
        if not self.x_errors:
            self.get_logger().info("No errors to plot.")
            return

        plt.figure(figsize=(12, 8))

        # Plot x error
        plt.subplot(3, 1, 1)
        plt.plot(self.x_errors, label="X Error")
        plt.xlabel("Time Steps")
        plt.ylabel("X Error (m)")
        plt.title("X Position Error Over Time")
        plt.legend()
        plt.grid()

        # Plot y error
        plt.subplot(3, 1, 2)
        plt.plot(self.y_errors, label="Y Error")
        plt.xlabel("Time Steps")
        plt.ylabel("Y Error (m)")
        plt.title("Y Position Error Over Time")
        plt.legend()
        plt.grid()

        # Plot theta error
        plt.subplot(3, 1, 3)
        plt.plot(self.theta_errors, label="Theta Error")
        plt.xlabel("Time Steps")
        plt.ylabel("Theta Error (radians)")
        plt.title("Orientation Error Over Time")
        plt.legend()
        plt.grid()

        plt.tight_layout()
        plt.show()


def main(args=None):
    rclpy.init(args=args)
    node = PoseDifferenceNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Shutting down.")
    finally:
        node.plot_errors()  # Plot errors after shutdown
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()

