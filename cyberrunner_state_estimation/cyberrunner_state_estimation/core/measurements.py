import cv2
import numpy as np
from rclpy.impl import rcutils_logger

from cyberrunner_state_estimation.core.detection import Detector
from cyberrunner_state_estimation.core.plate_pose import PlatePoseEstimator
from cyberrunner_state_estimation.utils.anim_3d import Anim3d
from cyberrunner_state_estimation.utils.divers import init_win_subimages


class Measurements:
    """
    A class for extracting measurements of the system state from the camera images.
    These measurements include the position of the ball in the maze frame and the angles of the board surface.
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
        The measurements calculated during this method must be accessed via methods such as:
        get_ball_position_in_maze(), get_plate_pose(), etc.

        Args :
            frame: np.ndarray, dim: (400, 640, 3)
                An image from the camera
            get_ball_subimg: bool
                Whether to generate a subimage centered on the ball as well
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
        # Alpha is the angle of the longer (horizontal) axis, around the -Y axis
        # Beta is the angle of the shorter (vertical) axis, around the +X axis
        alpha, beta = self.plate_pose.estimate_angles(corners_undistorted)
        self.plate_angles = (alpha, beta)

        # Compute the 3D position of the ball in the maze frame {m}
        self.ball_pos = self.ball_pos_backproject(
            ball_undistorted, self.plate_pose.K, self.plate_pose.T__C_M
        )

        # Generate the ball sub-image, if requested
        if get_ball_subimg:
            self.ball_subimg = self.ball_subimage_from_frame(frame)

        # Update our 3D animation(s)
        self.update_3d_anim(self.anim_3d_top)
        self.update_3d_anim(self.anim_3d_side)

    def ball_subimage_from_frame(self, frame):
        """
        Return a 64x64 sub-image from frame, centered on the ball

        Args :
            frame: np.ndarray, dim: (400, 640, 3)
                An image from the camera
        """
        # If we couldn't locate the ball, return an entirely black image
        if np.any(np.isnan(self.ball_pos)):
            return np.zeros((64, 64, 3), dtype=np.uint8)

        else:
            # Create a set of (homogeneous) coordinates in a 32mmx32mm square around the ball's location in the maze frame
            img_size = 64  # Image will be size x size pixels

            # Set up the lists of X- and Y-coordinates to include in the image
            x_coords = np.linspace(start=-32, stop=31, num=img_size) * 1.0e-3 + self.ball_pos[0]
            y_coords = np.linspace(start=32, stop=-31, num=img_size) * 1.0e-3 + self.ball_pos[1]

            # Create the homogeneous coordinates, each row is a point in the maze frame
            maze_coords = np.zeros((img_size ** 2, 4))
            maze_coords[:, 0] = np.tile(x_coords, img_size)  # X-coord
            maze_coords[:, 1] = np.repeat(y_coords, img_size)  # Y-coord
            maze_coords[:, 2] = self.ball_pos[2]  # Z-coord
            maze_coords[:, 3] = 1.0  # Homogeneous coordinates

            # Transform these points into 3D camera coordinates
            cam_coords = (self.plate_pose.T__C_M @ maze_coords.T).T[:, :3]

            # Transform these camera coordinates into image coordinates
            cam_coords[:, [0, 1]] = cam_coords[:, [1, 0]]  # Not sure why we need this line and the next!
            cam_coords[:, 2] *= -1  # Are the coords expected by world2cam() different???
            img_coords = self.plate_pose.o.world2cam(cam_coords)

            # Map these frame coordinates into a new image
            # Since we'll be indexing into a NumPy array, we must swap back to (row, col) convention
            img_coords = img_coords[:, ::-1].reshape(64, 64, 2).astype(np.float32)
            subimg = cv2.remap(frame, img_coords, map2=None, interpolation=cv2.INTER_LINEAR)

            # Return the sub-image
            return subimg

    # -----------------------------------------------
    # Measurement accessors
    # -----------------------------------------------
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
        Return the 3D position of the ball in the maze frame {m}.

        Returns:
            ball_pos: np.ndarray, dim: (3,)
                    3D position of the ball in the maze frame {m}.
                    NOTE: the z-coordinate of the ball in the maze frame is fixed and known
                    by assumption of constant contact with the maze: z__m_b = ball_radius.

        """
        return self.ball_pos

    def get_plate_pose(self):
        """
        Return the angles that describe the orientation of the maze frame {m} wrt the world frame {w}.

        Returns:
            Tuple(float, float)
                 (alpha, beta) around -Y and +X respectively, following the convention in the original paper
        """
        return self.plate_angles

    # -----------------------------------------------
    # Utilities
    # -----------------------------------------------

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
        """
        Create a mask for hiding the background of the camera images
        Args:
            frame: np.ndarray, dim: (400, 640, 3)
                An image from the camera
        """
        # Generate a list of all image coordinates
        h, w = frame.shape[:2]
        coords = np.array(list(np.ndindex((h, w))))

        # Get 3D camera frame coordinates for every pixel
        camera_points = self.plate_pose.o.cam2world(coords)

        # Again, I'm not sure why we need the following two lines
        # They appear to re-interpret the camera frame coordinates
        camera_points[:, [0, 1]] = camera_points[:, [1, 0]]
        camera_points[:, 2] *= -1

        # Transform these camera coordinates to world coordinates,
        # and project them all to the Z=0 plane in the world frame.
        world_vec = (self.plate_pose.T__W_C[:3, :3] @ camera_points.T).T    # Perform just the rotation first
        world_vec = world_vec / world_vec[:, 2:]           # Normalize all coordinates to have Z=1
        world_points = (                                   # Scale by the negative Z-translation and translate
            world_vec * (-self.plate_pose.T__W_C[2, -1])   # This will leave all Z-coordinates at 0
            + self.plate_pose.T__W_C[:3, -1]
        )

        # Determine which world points are within the bounds of the game
        mask = ((world_points[:, 0] >= -self.plate_pose.MARKER_DIAM)
              & (world_points[:, 0] <= self.plate_pose.L_EXT_INT_X + self.plate_pose.MARKER_DIAM)
              & (world_points[:, 1] >= -self.plate_pose.MARKER_DIAM)
              & (world_points[:, 1] <= self.plate_pose.L_EXT_INT_Y + self.plate_pose.MARKER_DIAM))

        # Create a mask for these points, with value 255 for all pixels to keep
        self.mask = 255 * mask.reshape(h, w, 1).astype(np.uint8)

    def ball_pos_backproject(self, ball_undistorted, K, T__C_M):
        """
        Compute the 3D position of the ball in the maze frame {m}.

        Args:
            ball_undistorted: np.ndarray, dim: (2,)
                The undistorted pixel coordinates of the ball in (row, col) convention
            K: np.ndarray, dim: (3, 3)
                The camera's intrinsic matrix
            T__C_M: np.ndarray, dim: (4, 4)
                The transform matrix from maze coordinates to camera coordinates

        Returns:
            x_M: np.ndarray, dim: (3,)
                 3d position of the ball in the maze frame {m}.
                 Note: the z-coordinate of the ball in maze frame is fixed and known
                 by assumption of constant contact with the maze: z__m_b = ball_radius.
        """
        # If the ball was not located in the image, return NaN's (unknown)
        if np.any(np.isnan(ball_undistorted)):
            return np.array([np.nan, np.nan, np.nan])

        v, u = ball_undistorted    # Convert from (row, col) to (u, v) convention
        d = PlatePoseEstimator.BALL_RAD

        # EXPLANATION:
        # We want to find the ball's position in maze coordinates, given its pixel coordinates in the image.
        # A good explanation of the math used here is available at:
        # https://docs.opencv.org/4.x/d9/d0c/group__calib3d.html
        # NOTES: Our 'K' matrix is referred to as A in the above description, and we're using the maze frame
        # as our "world" coordinates.

        # If p is the homogeneous representation of the image coordinates (i.e., [u, v, 1]),
        # and Pm is the homogeneous vector in maze coordinates,
        # the normal "forward projection" from maze coordinates to image coordinates would be accomplished by:
        #
        #          s p = K [R|t] Pm
        #
        # Here, s is an arbitrary scaling factor, and [R|t] is the rotation matrix and translation vector for
        # converting maze coordinates to camera coordinates. (This is just the normal transform matrix without
        # the final row of [0, 0, 0, 1])

        # In our case, we're given p, K, and [R|t], and need to solve for Pm.
        # Below, we refer to the product: K [R|t] as the matrix H for convenience.
        # However, we impose an additional constraint on the equations - we know the Z-coordinate of Pm,
        # which is fixed to the ball's radius (d), since we assume that the ball is in contact with the maze.
        # Thus, we want to fix Pm_z, and solve for Pm_x and Pm_y only.

        # Thus, if we define Pm as [Px, Py, d, 1] (a homogeneous vector), we have the equation:
        #
        #         H Pm = s p                 Recall: p = [u, v, 1]^T
        #
        # This system has 3 equations and 3 unknowns (Px, Py, and s).
        # To solve it, we can first use the last equation implied by this system to solve for s.
        # The last equation implied by this system is:
        #
        #         h_31*Px + h_32*Py + h33_d + h34 = s
        #
        # We can now substitute this expression for s on the right-hand side of the first two equations to eliminate s,
        # and solve the remaining two equations for the other unknowns: Px and Py
        # Multiplying out the expressions and gathering the terms for Px and Py on the left-hand side
        # (and moving all other terms only involving H, d, u, and v to the right-hand side),
        # we end up with a set of two equations in two unknowns (Px and Py) which we can solve using NumPy.

        # Define H = K [R|t], and get vars for its individual elements
        H = K @ T__C_M[:3, :]
        h_11, h_12, h_13, h_14 = H[0, :]
        h_21, h_22, h_23, h_24 = H[1, :]
        h_31, h_32, h_33, h_34 = H[2, :]

        # Form the 2x2 matrix A implied by the final two equations as described above:
        # NOTE: this only uses terms from the first-two columns of H, as expected
        A = np.array(
            [[u * h_31 - h_11, u * h_32 - h_12],
             [v * h_31 - h_21, v * h_32 - h_22]]
        )

        # Form the right-hand side of the system as described above
        # NOTE: the right-hand side only includes known terms: H, d, u, and v, as expected
        b = np.array(
            [
                d * h_13 + h_14 - d * u * h_33 - u * h_34,
                d * h_23 + h_24 - d * v * h_33 - v * h_34
            ]
        )

        # Solve this system for the 2 unknowns, which represent the ball's coordinates in the maze frame: Px and Py
        x = np.linalg.solve(A, b)

        # Return the 3D coordinates of the ball in the maze frame
        x_M = np.array([x[0], x[1], PlatePoseEstimator.BALL_RAD])
        return x_M

    def update_3d_anim(self, anim: Anim3d):
        # Ignore if the specified animation doesn't exist
        if anim is None:
            return

        # Transform the ball's position from maze coordinates to world coordinates
        Bw = (self.plate_pose.T__W_M @ np.append(self.ball_pos, 1))[:-1]

        # Update the ball position and corners
        anim.B__W = Bw
        anim.maze_corners__W = self.plate_pose.estimate_maze_corners__W()

        # Update the animation
        anim.update_anim()
