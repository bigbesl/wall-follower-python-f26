import time
import numpy as np
from mbot_bridge.api import MBot


def find_min_dist(ranges, thetas):
    """Finds the length and angle of the minimum ray in the scan.

    Make sure you ignore any rays with length 0! Those are invalid.

    Args:
        ranges (list): The length of each ray in the Lidar scan.
        thetas (list): The angle of each ray in the Lidar scan.

    Returns:
        tuple: The length and angle of the shortest ray in the Lidar scan.
    """
    min_dist, min_angle = None, None

    for dist, angle in zip(ranges, thetas):
        # Rays with zero range are invalid and must be skipped.
        if dist <= 0:
            continue
        if min_dist is None or dist < min_dist:
            min_dist, min_angle = dist, angle

    return min_dist, min_angle


def cross_product(v1, v2):
    """Compute the Cross Product between two vectors.

    Args:
        v1 (list): First vector of length 3.
        v2 (list): Second vector of length 3.

    Returns:
        list: The result of the cross product operation.
    """
    res = np.zeros(3)
    res[0] = v1[1] * v2[2] - v1[2] * v2[1]
    res[1] = v1[2] * v2[0] - v1[0] * v2[2]
    res[2] = v1[0] * v2[1] - v1[1] * v2[0]
    return res


robot = MBot()
setpoint = 0.5    # Desired distance from the wall (meters).
drive_speed = 0.3  # Forward speed along the wall (m/s).
kp = 1.0           # P-control gain for the distance correction.
max_correction = 0.5  # Cap on the correction speed (m/s).

try:
    while True:
        # Read the latest lidar scan.
        ranges, thetas = robot.read_lidar()

        # Find the nearest wall.
        min_dist, min_angle = find_min_dist(ranges, thetas)
        if min_dist is None:
            # No valid rays in this scan; stop and try again.
            robot.stop()
            time.sleep(0.1)
            continue

        # Unit vector pointing from the robot toward the wall.
        to_wall = np.array([np.cos(min_angle), np.sin(min_angle), 0.0])

        # Cross with z-hat to get a unit vector parallel to the wall,
        # pointing in the direction the robot should drive.
        along_wall = cross_product([0.0, 0.0, 1.0], to_wall)

        # P-control: positive error means we are too far from the wall,
        # so push toward it; negative error pushes away from it.
        error = min_dist - setpoint
        correction = np.clip(kp * error, -max_correction, max_correction)

        # Combine driving along the wall with the correction toward/away
        # from it, then send the velocity command.
        velocity = drive_speed * along_wall + correction * to_wall
        robot.drive(velocity[0], velocity[1], 0.0)

        # Sleep for a bit before reading a new scan.
        time.sleep(0.1)
except:
    # Catch any exception, including the user quitting, and stop the robot.
    robot.stop()
