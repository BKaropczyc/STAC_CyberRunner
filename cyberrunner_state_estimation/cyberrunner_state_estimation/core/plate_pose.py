import os
import cv2
import numpy as np
from ament_index_python.packages import get_package_share_directory
from cyberrunner_state_estimation.utils.ocam_model import OcamModel


class PlatePoseEstimator:
    """
    This class defines the World, Camera, and Maze frams, and the transforms for converting between them.
    It is also used to estimate the angles of playing board.
    """

    # Constants
    # All measurements are in meters
    L_EXT_INT_X = 0.317    # Length between the inner-edges of the exterior frame along the X-axis
    L_EXT_INT_Y = 0.2715   # Length between the inner-edges of the exterior frame along the Y-axis
    C2C_X = 0.2835         # Distance between (movable) corner circle centers along X axis
    C2C_Y = 0.2385         # Distance between (movable) corner circle centers along Y axis
    H_BORDERS = 0.022      # Height of the maze borders (from board surface to top of the game)

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
        dtype=np.float32,
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
        o = OcamModel(os.path.join(share, "calib_results_cyberrunner.txt"))
        self.o = o

        # Scale from 1920 resolution to 640
        o.scale(3)
        xc, yc = o.xc, o.yc

        # Define "camera intrinsic matrices" for OpenCV and Ocam
        self.f = 400                            # Focal length. Should be updated for your specific camera setup!
        self.K = np.array([[self.f, 0, yc],     # Ocam uses (row, col) convention for xc, yc, so we must swap them.
                           [0, self.f, xc],
                           [0, 0, 1]])
        self.K_ocam = np.array([[-self.f, 0, xc],   # Ocam uses a reversed Z-axis, so focal length is negative (???)
                                [0, -self.f, yc],
                                [0, 0, 1]])

        # Declare other instance vars
        self.T__W_M = None       # The World->Maze transform
        self.T__W_C = None       # The World->Camera transform

    def camera_localization(self, img_fix_pts):
        """
        Estimate the pose of camera {c} wrt the world frame {w}: T__W_C (also noted T^W_C).

        Args:
           img_fix_pts: np.ndarray, dim: (4,2)
                        the raw image coordinates of the four fixed reference dots of the external frame of the labyrinth
                        in (x,y) = (line, column) convention.
        """
        # Undistort the points using our camera model
        img_points_fixed_corners_undist = self.undistort_points(img_fix_pts)  # (x,y)

        # Get the pose of the world frame {w} wrt the camera frame {c}
        T__C_W = self.get_pose_T__C_P(
            PlatePoseEstimator.FIXED_CORNERS_WORLD_COORDS,
            img_points_fixed_corners_undist
        )

        # The inverse of this transform is the pose of the camera {c} wrt the world frame {w}
        self.T__W_C = self.invert_transform(T__C_W)

    def estimate_angles(self, corners_undist, deg=False):
        """
        Compute the angles that describe the orientation of the maze frame {m} wrt to the world frame {w}.

        Args :
            corners_undist: np.ndarray, dim: (4,2)
                            undistorted image coordinates of the maze corners dots in (x,y) = (line, column) convention.
        Returns :
            alpha: float
                    angle around -Y axis  (following the convention given in the original paper)
            beta: float
                    angle around +X axis
        """
        # Get the World->Maze transformation matrix
        self.T__W_M = self.get_maze_pose_in_world(corners_undist)

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
                           undistorted image coordinates of the maze corners dots in (x,y) = (line, column) convention.

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

    def undistort_points(self, img_points_raw: np.ndarray):  # (x,y)
        """
        Undistort the points using the camera calibration data via cam2world.

        Args :
            img_points_raw:    np.ndarray, dim: (N,2)
                               image coordinates of the raw points in (x,y) = (line, column) convention.
        Returns :
            img_point_undist:  np.ndarray, dim: (N,2)
                               image coordinates of the undistorted points in (x,y) = (line, column) convention.

        """
        P_w = self.o.cam2world(img_points_raw).T  # dim: (3, N)
        pt_undist = self.K_ocam @ P_w  # dim: (3, N)
        pt_undist = pt_undist / pt_undist[2, :]
        img_point_undist = pt_undist.T[:, :2]
        return img_point_undist

    def get_pose_T__C_P(
        self, model_points: np.ndarray, img_points: np.ndarray
    ):
        """
        Compute the pose of the frame {p} in which model points are expressed wrt to the camera frame {c}.

        Args :
            model_points: np.ndarray, dim: (4,3)
                           3D coordinates of the points in their frame {p}.
            img_points: np.ndarray, dim: (4,2)
                        undistorted image coordinates of the corresponding points in (x,y) = (line, column) convention.

        Returns :
            T__C_P: np.ndarray, dim: (4,4)
                  pose in SE(3) of the frame {p} in which model points are expressed wrt to the camera frame {c}.
        """
        # Convert the points to opencv convention: (u,v) = (column, line)
        img_points = np.flip(
            img_points, axis=1
        )

        # Get the rotation and translation vectors for the C->P transform
        _, rotation_vec, translation_vec = cv2.solvePnP(
            model_points, img_points, self.K, None, flags=cv2.SOLVEPNP_ITERATIVE
        )

        # Convert the rotation vector into a standard 3x3 rotation matrix
        rotation_mat, _ = cv2.Rodrigues(rotation_vec)

        # Return the 4x4 homogeneous transformation matrix
        T__C_P = np.hstack((rotation_mat, translation_vec))
        T__C_P = np.vstack((T__C_P, np.array([0, 0, 0, 1])))
        return T__C_P

    def invert_transform(self, T):
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
