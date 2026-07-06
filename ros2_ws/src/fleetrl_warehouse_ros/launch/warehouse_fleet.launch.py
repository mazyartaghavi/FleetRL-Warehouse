from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import AppendEnvironmentVariable, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node


def generate_launch_description():
    package_share = Path(get_package_share_directory("fleetrl_warehouse_ros"))
    ros_gz_share = Path(get_package_share_directory("ros_gz_sim"))
    world = package_share / "worlds" / "warehouse.sdf"
    model_path = package_share / "models"

    actions = [
        AppendEnvironmentVariable("GZ_SIM_RESOURCE_PATH", str(model_path)),
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(str(ros_gz_share / "launch" / "gz_sim.launch.py")),
            launch_arguments={"gz_args": f"-r -v 2 {world}"}.items(),
        ),
        Node(
            package="fleetrl_warehouse_ros",
            executable="fleet_coordinator",
            name="fleet_coordinator",
            output="screen",
        ),
    ]

    for robot_id in range(3):
        name = f"robot_{robot_id}"
        bridge_args = [
            f"/model/{name}/cmd_vel@geometry_msgs/msg/Twist@gz.msgs.Twist",
            f"/model/{name}/odom@nav_msgs/msg/Odometry@gz.msgs.Odometry",
            f"/world/fleetrl_warehouse/model/{name}/link/base_link/sensor/lidar/scan@sensor_msgs/msg/LaserScan@gz.msgs.LaserScan",
        ]
        actions.extend(
            [
                Node(
                    package="ros_gz_bridge",
                    executable="parameter_bridge",
                    name=f"{name}_bridge",
                    arguments=bridge_args,
                    remappings=[
                        (f"/model/{name}/cmd_vel", f"/{name}/cmd_vel"),
                        (f"/model/{name}/odom", f"/{name}/odom"),
                        (
                            f"/world/fleetrl_warehouse/model/{name}/link/base_link/sensor/lidar/scan",
                            f"/{name}/scan",
                        ),
                    ],
                    output="screen",
                ),
                Node(
                    package="fleetrl_warehouse_ros",
                    executable="policy_executor",
                    name=f"{name}_policy",
                    parameters=[{"robot_name": name, "robot_id": robot_id}],
                    output="screen",
                ),
                Node(
                    package="fleetrl_warehouse_ros",
                    executable="safety_supervisor",
                    name=f"{name}_safety",
                    parameters=[{"robot_name": name}],
                    output="screen",
                ),
            ]
        )
    return LaunchDescription(actions)
