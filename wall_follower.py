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

SETPOINT = 0.5          # Desired distance from the wall (meters).
DRIVE_SPEED = 0.25      # Speed along the wall (m/s).
CORRECTION_SPEED = 0.1  # Bang-bang correction speed toward/away from the wall (m/s).

# The Lidar reports angles measured clockwise, which is the opposite of the
# robot body frame's right-hand-rule convention. Set to 1.0 if the printed
# min_angle is positive when the wall is physically on the robot's left.
LIDAR_ANGLE_SIGN = -1.0

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

        angle = LIDAR_ANGLE_SIGN * min_angle
        print(f"dist: {min_dist:.2f} m   angle: {np.degrees(angle):6.1f} deg")

        # Unit vector pointing from the robot toward the nearest point on the
        # wall. The shortest ray is perpendicular to the wall, so this is the
        # wall normal.
        to_wall = np.array([np.cos(angle), np.sin(angle), 0.0])

        # Cross z-hat with the normal to get a unit vector parallel to the
        # wall. This ordering drives forward when the wall is on the right, so
        # the robot follows walls on its right-hand side.
        along_wall = cross_product([0.0, 0.0, 1.0], to_wall)

        # Bang-bang control: the correction is always full magnitude, only its
        # direction switches depending on which side of the setpoint we're on.
        if min_dist > SETPOINT:
            correction = CORRECTION_SPEED       # Too far, steer into the wall.
        else:
            correction = -CORRECTION_SPEED      # Too close, steer away.

        # Combine driving along the wall with the correction toward/away from
        # it, then send the velocity command.
        velocity = DRIVE_SPEED * along_wall + correction * to_wall
        robot.drive(velocity[0], velocity[1], 0.0)

        # Sleep for a bit before reading a new scan.
        time.sleep(0.05)
except KeyboardInterrupt:
    pass
finally:
    # Always stop the robot on the way out, however the loop ended.
    robot.stop()
