import cv2
import numpy as np
from rclpy.impl import rcutils_logger

from cyberrunner_state_estimation.core.detection import Detector
from cyberrunner_state_estimation.core.plate_pose import PlatePoseEstimator
from cyberrunner_state_estimation.utils.anim_3d import Anim3d
from cyberrunner_state_estimation.utils.divers import init_win_subimages


class Measurements:
    """
    To be written...
    """
    def __init__(
        self, markers, show_3d_anim=True, viewpoint="side", show_subimage_masks=False
    ):
        # Create our object detectors and pose estimator
        # The first 4 rows of markers are the positions of the outer (static) corner markers (LL, LR, UR, UL)
        # The last 4 rows of markers are the positions of the inner (movable) corner markers
        self.detector = Detector(markers[4:], show_subimage_masks=show_subimage_masks)
        self.detector_fixed_points = Detector(markers[:4], markers_are_static=True, show_subimage_masks=show_subimage_masks)
        self.plate_pose = PlatePoseEstimator()

        # Initialize our measurements
        self.plate_angles = (None, None)    # The inclination angles of the playing surface (alpha, beta)
        self.ball_pos = None                # The 3D position of the ball in the maze frame {m}
        self.ball_img_coords = None         # The pixel coordinates of the ball in the image frame {c}
        self.ball_subimg = None             # A 64x64 pixel image centered on the ball in the maze frame {m}

        # Initialize our 3D animations
        self.anim_3d_top = None
        self.anim_3d_side = None
        if show_3d_anim:
            if "side" in viewpoint:
                self.anim_3d_side = Anim3d(viewpoint="side")
            if "top" in viewpoint:
                self.anim_3d_top = Anim3d(viewpoint="top")

        # Create our subimage mask windows
        if show_subimage_masks:
            init_win_subimages()      # There has GOT to be a better place to do this...

    def process_frame(
        self,
        frame,
        get_ball_subimg=False
    ):
        """
        Process the frame to compute the angles of the plate and the position of the ball in the maze frame {m}.

        Args :
            frame: np.ndarray, dim: (400, 640, 3)
        """
        # Perform one-time initializations
        if self.plate_pose.T__W_C is None:
            # Compute the camera->world transform, T__W_C
            self.camera_localization(frame)

            # Create a mask to hide the background, leaving only the playing area
            self.create_mask(frame)

            # Initialize our 3D animations with this transform
            if self.anim_3d_top is not None:
                self.anim_3d_top.init_3d_anim(self.plate_pose.T__W_C)
            if self.anim_3d_side is not None:
                self.anim_3d_side.init_3d_anim(self.plate_pose.T__W_C)

        # Mask out the background of the image
        frame = cv2.bitwise_and(frame, frame, mask=self.mask)

        # Determine the coordinates of the (moveable) corners and the ball within the image
        # All coordinates are in (row, column) format
        corners_img_coords, ball_img_coords = self.detector.process_frame(frame)
        self.ball_img_coords = ball_img_coords

        # Undistort all of these points using the camera calibration data
        # These coordinates are still wrt the camera imaga
        raw_pts = np.vstack((corners_img_coords, ball_img_coords))
        undist_pts = self.plate_pose.undistort_points(raw_pts)
        corners_undistorted = undist_pts[:4, :]
        ball_undistorted = undist_pts[4, :]

        # Estimate the angles of the playing surface from the locations of the corners in the image
        # Alpha is the angle of the longer (horizontal) axis
        # Beta is the angle of the shorter (vertical) axis
        alpha, beta = self.plate_pose.estimate_anglesXY(corners_undistorted)
        self.plate_angles = (alpha, beta)

        # Compute the 3D position of the ball in the maze frame {m}
        self.ball_pos = self.ball_pos_backproject(
            ball_undistorted, self.plate_pose.K, self.plate_pose.T__C_M
        )

        if get_ball_subimg:  # TODO: make function
            # Return a 64x64 subimage centered on the ball

            # If we couldn't locate the ball, return an entirely black image
            if np.any(np.isnan(self.ball_pos)):
                self.ball_subimg = np.zeros((64, 64, 3), dtype=np.uint8)
            else:  # TODO clean up and optimize
                # Create collection of coordinates in a 32mmx32mm square around the ball's location in the maze frame
                points_board = np.zeros((64 * 64, 4))
                points_board[:, -1] = 1.0
                points_board[:, :2] = (
                    1.0e-3 * np.mgrid[-32:32, -32:32][::-1].reshape(2, -1).transpose()
                )
                points_board[:, 1] *= -1
                points_board[:, :3] += self.ball_pos

                # Transform those points into camera coordinates
                points_cam = (self.plate_pose.T__C_M @ points_board.T).T[:, :3]
                points_cam[:, :2] = points_cam[:, [1, 0]]
                points_cam[:, 2] *= -1
                points_cam = self.plate_pose.o.world2cam(points_cam)   # Undo camera distortion...
                points_cam = points_cam.reshape(64, 64, 2).astype(np.float32)

                self.ball_subimg = cv2.remap(
                    frame, points_cam[..., 1], points_cam[..., 0], cv2.INTER_LINEAR
                )

        # Update our 3D animation(s)
        if self.anim_3d_top is not None:
            self.update_3d_anim_top()
        if self.anim_3d_side is not None:
            self.update_3d_anim_side()

    def get_ball_subimg(self):
        return self.ball_subimg

    def get_ball_coordinates(self):
        """
        Return the pixel coordinates of the ball in the image frame {c}.

        Returns:
            ball_pos: np.ndarray, dim: (2,)
                    2d position of the ball in the image frame.

        """
        return self.ball_img_coords

    def get_ball_position_in_maze(self):
        """
        Return the position of the ball in the maze frame {m}.

        Returns:
            ball_pos: np.ndarray, dim: (3,)
                    3d position of the ball in the maze frame {m}.
                    note: the z-coordinate of the ball in maze frame is fixed and known
                    by assumption of constant contact with the maze: z__m_b = ball_radius.

        """
        return self.ball_pos

    def get_plate_pose(self):
        """
        Return the angles (Euler YXZ) that describe the orientation of the maze frame {m} wrt the world frame {w}.

        Returns:
            ball_pos: Tuple(float, float)
                      (alpha, beta) around X and Y respectively
        """
        return self.plate_angles

    def camera_localization(self, frame):
        """
        Compute the pose of the camera {c} wrt to the world frame {w} : T__W_C.
        """
        # Get the positions of the 4 fixed markers in the image
        fix_pts = self.detector_fixed_points.detect_corners(frame)

        # Make sure all corners were found, otherwise we can't continue
        corners_found = self.detector_fixed_points.corners_found
        if not all(corners_found):
            log_message = f"Camera localization failed: Could not detect all outer corners.\nCorners found: {corners_found}"
            logger = rcutils_logger.RcutilsLogger(name="Measurements")
            logger.fatal(log_message)

            # Display the frame that we couldn't use for camera localization
            cv2.imshow("Camera Localization Failure", frame)
            cv2.waitKey(10 * 1000)

            exit(1)

        # Let the inner corner markers Detector object know where these are as well, so it can hide them as necessary
        self.detector.fixed_corners = fix_pts

        # Estimate the pose of camera wrt the world frame: T__W_C
        self.plate_pose.camera_localization(fix_pts)

    def create_mask(self, frame):
        h, w = frame.shape[:2]
        coords = np.mgrid[0:h, 0:w].transpose(1, 2, 0).reshape(-1, 2)
        camera_points = self.plate_pose.o.cam2world(coords)[:, [1, 0, 2]]
        camera_points[:, 2] *= -1
        world_vec = (self.plate_pose.T__W_C[:3, :3] @ camera_points.T).T
        world_vec = world_vec / world_vec[:, 2:]
        world_points = (
            world_vec * (-self.plate_pose.T__W_C[2, -1])
            + self.plate_pose.T__W_C[:3, -1]
        )
        mask = (
            (world_points[:, 0] >= -(2.0 * self.plate_pose.r))
            & (
                world_points[:, 0]
                <= self.plate_pose.L_EXT_INT_X + 2.0 * self.plate_pose.r
            )
            & (world_points[:, 1] >= -(2.0 * self.plate_pose.r))
            & (
                world_points[:, 1]
                <= self.plate_pose.L_EXT_INT_Y + 2.0 * self.plate_pose.r
            )
        )
        self.mask = 255 * mask.reshape(h, w, 1).astype(np.uint8)

    def ball_pos_backproject(self, ball_undistorted, K, T__C_M):
        """
        Compute the 3d position of the ball in the maze frame {m}.

        Returns:
            x_M: np.ndarray, dim: (3,)
                 3d position of the ball in the maze frame {m}.
                 note: the z-coordinate of the ball in maze frame is fixed and known
                 by assumption of constant contact with the maze: z__m_b = ball_radius.
        """
        # If the ball was not located in the image, return NaN's (unknown)
        if np.any(np.isnan(ball_undistorted)):
            return np.array([np.nan, np.nan, np.nan])

        d = PlatePoseEstimator.R_BALL
        v, u = ball_undistorted

        H = K @ T__C_M[:3, :]
        h_11, h_12, h_13, h_14 = H[0, :]
        h_21, h_22, h_23, h_24 = H[1, :]
        h_31, h_32, h_33, h_34 = H[2, :]

        # Solve the equation Ax=b
        # A is a 2x2 matrix, b is a 2x1 vector
        A = np.array(
            [[u * h_31 - h_11, u * h_32 - h_12],
             [v * h_31 - h_21, v * h_32 - h_22]]
        )
        b = np.array(
            [
                d * h_13 + h_14 - d * u * h_33 - u * h_34,
                d * h_23 + h_24 - d * v * h_33 - v * h_34,
            ]
        )
        x = np.linalg.solve(A, b)

        # Return the coordinates of the ball in the maze frame
        x_M = np.array([x[0], x[1], PlatePoseEstimator.R_BALL])
        return x_M

    def update_3d_anim_top(self):
        self.anim_3d_top.B__W = (
            self.plate_pose.T__W_M @ np.hstack((self.ball_pos, np.array([1])))
        )[:-1]
        self.anim_3d_top.maze_corners__W = self.plate_pose.estimate_maze_corners__W()
        self.anim_3d_top.update_anim()

    def update_3d_anim_side(self):
        self.anim_3d_side.B__W = (
            self.plate_pose.T__W_M @ np.hstack((self.ball_pos, np.array([1])))
        )[:-1]
        self.anim_3d_side.maze_corners__W = self.plate_pose.estimate_maze_corners__W()
        self.anim_3d_side.update_anim()
