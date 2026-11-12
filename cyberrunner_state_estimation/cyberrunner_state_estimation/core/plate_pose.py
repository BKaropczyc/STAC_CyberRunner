import os
import cv2
import numpy as np
from ament_index_python.packages import get_package_share_directory


class PlatePoseEstimator:
    """
    This class defines the World, Camera, and Maze frames, and the transforms for converting between them.
    It is also used to estimate the angles of the playing board.
    """

    # Constants
    # All measurements are in meters
    L_EXT_INT_X = 0.317    # Length between the inner-edges of the exterior frame along the X-axis
    L_EXT_INT_Y = 0.2706   # Length between the inner-edges of the exterior frame along the Y-axis
    C2C_X = 0.2812         # Distance between (movable) corner circle centers along X axis
    C2C_Y = 0.2367         # Distance between (movable) corner circle centers along Y axis
    H_BORDERS = 0.0215     # Height of the maze borders (from board surface to top of the game)

    MARKER_DIAM = 0.008             # Diameter of the cornet marker circles (dot stickers)
    MARKER_RAD = MARKER_DIAM / 2    # Radius of the corner marker circles (dot stickers)
    BALL_RAD = 0.012 / 2            # Radius of the ball
    EDGE_WIDTH = 0.0075             # Width of the frame edges

    # These are the 3D coordinates of the four fixed corner markers (center of circles) in World coordinates,
    # and they are used to define the World frame (W).
    # The World frame's origin is at the top of lower-left corner of the inner edge of the outer frame.
    # The X-axis points to the right along the horizontal axis of the game
    # The Y-axis points up along the vertical axis of the game
    # The Z-axis points up, perpendicular to the surface the game is sitting on
    FIXED_CORNERS_WORLD_COORDS = np.array(
        [
            (-MARKER_RAD, 0.05, 0),  # Corner 1
            (L_EXT_INT_X + MARKER_RAD, 0.05, 0),  # Corner 2
            (L_EXT_INT_X + MARKER_RAD, L_EXT_INT_Y - 0.05, 0),  # Corner 3
            (-MARKER_RAD, L_EXT_INT_Y - 0.05, 0)  # Corner 4
        ],
        dtype=np.float32
    )

    # These are the 3D coordinates of the four corner markers (center of circles) in Maze coordinates,
    # and they are used to define the Maze frame (M).
    # The Maze frame's origin is at the center of the board surface
    # The X-axis points to the right along the horizontal axis of the board surface
    # The Y-axis points up along the vertical axis of the board surface
    # The Z-axis is perpendicular to the game board surface, tilting with the board as it moves
    CORNERS_MAZE_COORDS = np.array(
        [
            [-C2C_X / 2, -C2C_Y / 2, H_BORDERS],  # Corner 1 (dl)
            [+C2C_X / 2, -C2C_Y / 2, H_BORDERS],  # Corner 2 (dr)
            [+C2C_X / 2, +C2C_Y / 2, H_BORDERS],  # Corner 3 (ur)
            [-C2C_X / 2, +C2C_Y / 2, H_BORDERS]   # Corner 4 (ul)
        ],
        dtype=np.float32
    )

    # Similar to above, but don't include the (half-width of the) board frame in the measurements.
    # This is only used for 3D plotting
    MAZE_CORNERS__M = np.array(
        [
            [-C2C_X / 2 + EDGE_WIDTH / 2, -C2C_Y / 2 + EDGE_WIDTH / 2, 0],
            [+C2C_X / 2 - EDGE_WIDTH / 2, -C2C_Y / 2 + EDGE_WIDTH / 2, 0],
            [+C2C_X / 2 - EDGE_WIDTH / 2, +C2C_Y / 2 - EDGE_WIDTH / 2, 0],
            [-C2C_X / 2 + EDGE_WIDTH / 2, +C2C_Y / 2 - EDGE_WIDTH / 2, 0]
        ]
    )

    def __init__(self):
        # Read in our camera calibration data
        share = get_package_share_directory("cyberrunner_state_estimation")
        calib_results_file = os.path.join(share, "calib_results_cyberrunner.txt")
        with open(calib_results_file, 'r') as file:
            lines = file.readlines()

        # Remove all empty and commented lines
        lines = [l for l in lines if len(l.strip()) > 0 and not l.startswith("#")]

        # There should be exactly 4 data lines
        assert len(lines) == 4, "Camera calibration file contents is invalid (wrong number of lines)"

        # Extract the calibration data
        img_w, img_h = (int(i) for i in lines[0].strip().split())
        fx, fy = (float(i) for i in lines[1].strip().split())
        cx, cy = (float(i) for i in lines[2].strip().split())
        dist_coeff = np.array([float(i) for i in lines[3].strip().split()])

        # Form the (unscaled) camera matrix (i.e., intrinsic parameter matrix)
        camera_matrix = np.array([[fx, 0,  cx],
                                  [0,  fy, cy],
                                  [0,  0,  1]])

        # Scale the camera_matrix down to a horizontal resolution of 640
        camera_matrix[:2] /= (img_w // 640)

        # Add 20 to cy to compensate for the 20px border we add (???)
        camera_matrix[1, 2] += 20

        # Store these camera parameters as instance vars
        self.K = camera_matrix
        self.dist_coeff = dist_coeff

        # Declare other instance vars for transformations
        self.T__W_M = None       # The World->Maze transform
        self.T__W_C = None       # The World->Camera transform
        self.T__C_M = None       # The Camera->Maze transform

    def camera_localization(self, img_fix_pts):
        """
        Estimate the pose of camera {c} wrt the world frame {w}: T__W_C (also noted T^W_C).

        Args:
           img_fix_pts: np.ndarray, dim: (4,2)
                        the raw image coordinates of the four fixed reference dots of the external frame of the labyrinth
                        in (u, v) convention.
        """
        # Get the pose of the world frame {w} wrt the camera frame {c}
        T__C_W = self.get_pose_T__C_P(
            PlatePoseEstimator.FIXED_CORNERS_WORLD_COORDS,
            img_fix_pts
        )

        # The inverse of this transform is the pose of the camera {c} wrt the world frame {w}
        self.T__W_C = self.invert_transform(T__C_W)

    def estimate_angles(self, corners, deg=False):
        """
        Compute the angles that describe the orientation of the maze frame {m} wrt to the world frame {w}.

        Args :
            corners: np.ndarray, dim: (4,2)
                 Image coordinates of the maze corners dots in (u, v) convention.
            deg: bool
                Whether to return the angles in degrees (True) or radians (False)

        Returns :
            alpha: float
                    angle around -Y axis  (following the convention given in the original paper)
            beta: float
                    angle around +X axis
        """
        # Get the World->Maze transformation matrix
        self.T__W_M = self.get_maze_pose_in_world(corners)

        # Extract the rotation matrix portion
        R = self.T__W_M[:3, :3]

        # Compute the angles from the entries in the rotation matrix
        alpha = np.arctan2(-R[0, 2], R[2, 2])  # around -Y
        beta = np.arctan2(-R[1, 2], R[2, 2])   # around +X

        # Convert to degrees, if desired
        if deg:
            alpha = np.rad2deg(alpha)
            beta = np.rad2deg(beta)

        # Return our results
        return alpha, beta

    def get_maze_pose_in_world(self, image_points):
        """
        Compute the pose of the maze {m} wrt to the world frame {w}.

        Args :
            image_points: np.ndarray, dim: (4,2)
                Image coordinates of the maze corner dots in (u, v) convention.

        Returns :
           T__W_M: np.ndarray, dim: (4,4)
                The SE(3) transform matrix describing the pose of the maze frame wrt the world frame
        """
        # Get the current Camera->Maze transformation matrix
        T__C_M = self.get_pose_T__C_P(PlatePoseEstimator.CORNERS_MAZE_COORDS, image_points)
        self.T__C_M = T__C_M

        # Calculate and return the World->Maze transformation matrix via composition
        T__W_M = self.T__W_C @ T__C_M
        return T__W_M

    # --------------------------------------------------------------------------------------------------------
    # Utilities
    # --------------------------------------------------------------------------------------------------------

    def image_to_camera_coords(self, img_coords: np.ndarray):
        """
        Convert 2D image pixel coordinates to 3D camera coordinates, accounting for any lens distortion.
        NOTE: The camera coordinates returned are "normalized" to have a Z-axis value of 1.
        Points at other depths can easily be formed by multiplying the returned point(s) by the desired depth.

        Args:
            img_coords: ndarray, dim: (2,) or (N, 2)
                The 2D image coordinates to convert to camera coordinates
                If 1D, img_coords is expected to be in (u, v) convention
                If 2D, each row of img_coords is expected to be in (u, v) convention

        Returns:
            An ndarray of dim (3, ) or (N, 3) containing normalized 3D camera coordinates
            If img_coords is 1D, result will be: [Xz, Yc, Zc=1]
            If img_coords is 2D, each row of the result will be: [Xc, Yc, Zc=1]
        """
        # Convert img_coords to normalized camera coordinates
        cam_coords = cv2.undistortPoints(img_coords, self.K, self.dist_coeff).squeeze()

        # Add a Z-coordinate of 1 (normalized)
        if cam_coords.ndim == 1:
            cam_coords = np.append(cam_coords, 1)
        else:
            cam_coords = np.column_stack((cam_coords, np.ones(len(cam_coords))))

        # Return the camera coordinates
        return cam_coords

    def maze_to_image_coords(self, maze_coords: np.ndarray):
        """
        Convert 3D coordinates in the "maze" frame to 2D image coordinates in (u, v) convention.
        The image coordinates returned are with respect to the "distorted" (uncorrected) image.
        This is useful to find where items in the maze frame appear in the image.

        Args:
            maze_coords: ndarray, dim:(N, 3)
                3D coordinates in the maze frame. Each row is [Xm, Yz, Zm]

        Returns:
            An ndarray of dimension (N, 2) containing image coordinates in (u, v) convention.
        """
        # Get the Camera->Maze transform as rotation and translation vectors
        rvec, _ = cv2.Rodrigues(self.T__C_M[:3, :3])
        tvec = self.T__C_M[:3, 3]

        # Project the maze coordinates into image coordinates
        img_coords = cv2.projectPoints(maze_coords,
                                       rvec=rvec,
                                       tvec=tvec,
                                       cameraMatrix=self.K,
                                       distCoeffs=self.dist_coeff)[0].squeeze()

        # Return the image coordinates
        return img_coords

    def get_pose_T__C_P(
        self, model_points: np.ndarray, img_points: np.ndarray
    ):
        """
        Compute the pose of the frame {p} in which model points are expressed wrt to the camera frame {c}.

        Args :
            model_points: np.ndarray, dim: (4,3)
                3D coordinates of the points in their frame {p}.
            img_points: np.ndarray, dim: (4,2)
                Image coordinates of the corresponding points in (u, v) convention.

        Returns :
            T__C_P: np.ndarray, dim: (4,4)
                  pose in SE(3) of the frame {p} in which model points are expressed wrt to the camera frame {c}.
        """
        # Get the rotation and translation vectors for the C->P transform
        _, rotation_vec, translation_vec = cv2.solvePnP(
            model_points, img_points, self.K, self.dist_coeff, flags=cv2.SOLVEPNP_ITERATIVE
        )

        # Convert the rotation vector into a standard 3x3 rotation matrix
        rotation_mat, _ = cv2.Rodrigues(rotation_vec)

        # Return the 4x4 homogeneous transformation matrix
        T__C_P = np.hstack((rotation_mat, translation_vec))
        T__C_P = np.vstack((T__C_P, np.array([0, 0, 0, 1])))
        return T__C_P

    @staticmethod
    def invert_transform(T):
        """
        Compute the inverse of the transform matrix T [4x4] in SE(3).

        Args :
            T: np.ndarray, dim: (4,4)
               transformation matrix in SE(3).
        Returns :
           T_inv: np.ndarray, dim: (4,4)
                  inverse of the matrix T in SE(3).
        """
        # Extract the rotation matrix and translation vector
        R = T[:3, :3]
        t = np.expand_dims(T[:3, -1], axis=1)

        # Compose the inverse transformation matrix using the standard formula
        T_inv = np.hstack((R.T, -R.T @ t))
        T_inv = np.vstack((T_inv, np.array([0, 0, 0, 1])))

        # Return the inverse transform
        return T_inv

    def estimate_maze_corners__W(self):  # used only for plotting
        """
        Estimate the coordinates of the corners of the maze (not the detection dots but the real corners of maze,
        i.e. the limits maze) in the world frame {w}. Useful for plotting only.

        Returns :
            Ps__W:  np.ndarray, dim: (4,3)
                    coordinates of the corners of the maze in the world frame {w}.

        """
        Ps__M = PlatePoseEstimator.MAZE_CORNERS__M.T
        Ps__W = (self.T__W_M @ np.vstack((Ps__M, np.ones(4))))[:-1, :]
        Ps__W = Ps__W.T
        return Ps__W
