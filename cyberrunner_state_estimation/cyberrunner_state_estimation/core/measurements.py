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

    # DEBUGGING OPTIONS:
    # Set the following options to True to enable additional debugging output
    SHOW_UNDISTORTED_FRAME = False   # Show the frame with the distortion corrected
    SHOW_PATH = False      # Show the maze path for calculating progress superimposed on the frame being processed
    SHOW_BALL_SUBIMAGE = False   # Show the 64x64 pixel subimage centered on the ball

    def __init__(
        self, markers, show_3d_anim=True, viewpoint="side", show_subimage_masks=False
    ):
        # Create our object detectors and pose estimator
        # The first 4 rows of markers are the positions of the outer (static) corner markers (LL, LR, UR, UL)
        # The last 4 rows of markers are the positions of the inner (movable) corner markers
        self.detector = Detector(markers[4:], show_subimage_masks=show_subimage_masks)
        self.detector_fixed_points = Detector(markers[:4], markers_are_static=True, show_subimage_masks=show_subimage_masks)
        self.plate_pose = PlatePoseEstimator()

        # Prepare to localize our camera
        self.camera_localized = False
        self.camera_localization_data = []
        self.mask = None

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

        # Debugging options
        if Measurements.SHOW_UNDISTORTED_FRAME:
            # Calculate the maps to undistort the image
            map1, map2 = cv2.initUndistortRectifyMap(self.plate_pose.K,
                                                     self.plate_pose.dist_coeff,
                                                     R=None,
                                                     newCameraMatrix=None,
                                                     size=(640, 400),
                                                     m1type=cv2.CV_32FC1)
            self.undistort_maps = (map1, map2)

        if Measurements.SHOW_PATH:
            # Load the maze path waypoint coordinates
            from cyberrunner_dreamer.cyberrunner_layout import cyberrunner_hard_layout
            waypoints = cyberrunner_hard_layout['waypoints']

            # Convert the waypoints to "maze" coordinates, with the origin in the center of the board
            center = np.array([0.276, 0.231]) / 2.0
            maze_pts = waypoints - center

            # Create full coordinates for the waypoints, each row is a 3D point in the maze frame
            path_coords = np.zeros((len(maze_pts), 3))
            path_coords[:, :2] = maze_pts
            path_coords[:, 2] = 0     # Visualize the path sitting on the playing surface, Z=0
            self.path_coords = path_coords

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
        if self.mask is None:
            # Create a mask to hide the background, leaving only the playing area
            self.create_mask(frame)

            # Initialize our 3D animations
            if self.anim_3d_top is not None:
                self.anim_3d_top.init_3d_anim(self.plate_pose.T__W_C)
            if self.anim_3d_side is not None:
                self.anim_3d_side.init_3d_anim(self.plate_pose.T__W_C)

        if Measurements.SHOW_UNDISTORTED_FRAME:
            # Display the undistorted frame
            map1, map2 = self.undistort_maps
            undistorted_img = cv2.remap(frame, map1, map2, interpolation=cv2.INTER_LINEAR)
            cv2.imshow("Undistorted Frame", undistorted_img)
            cv2.waitKey(1)

        # Mask out the background of the image
        frame = cv2.bitwise_and(frame, frame, mask=self.mask)

        # Determine the coordinates of the (moveable) corners and the ball within the image
        # All coordinates are in (u, v) convention
        corners_img_coords, ball_img_coords = self.detector.process_frame(frame)
        self.ball_img_coords = ball_img_coords

        # Estimate the angles of the playing surface from the locations of the corners in the image
        # Alpha is the angle of the longer (horizontal) axis, around the -Y axis
        # Beta is the angle of the shorter (vertical) axis, around the +X axis
        alpha, beta = self.plate_pose.estimate_angles(corners_img_coords)
        self.plate_angles = (alpha, beta)

        # Compute the 3D position of the ball in the maze frame {m}
        self.ball_pos = self.ball_pos_backproject(ball_img_coords)

        # Generate the ball sub-image, if requested
        if get_ball_subimg:
            self.ball_subimg = self.ball_subimage_from_frame(frame)
            if Measurements.SHOW_BALL_SUBIMAGE:
                cv2.imshow("Ball Subimage", self.ball_subimg)
                cv2.waitKey(1)

        # Update our 3D animation(s)
        self.update_3d_anim(self.anim_3d_top)
        self.update_3d_anim(self.anim_3d_side)

        if Measurements.SHOW_PATH:
            # Get image coordinates for the path waypoints
            img_coords = self.plate_pose.maze_to_image_coords(self.path_coords)

            # Use sub-pixel precision to draw the maze path
            shift_bits = 4
            img_coords = (img_coords * 2 ** shift_bits).astype(int)

            # Draw the maze path superimposed on a copy of the frame
            path_frame = frame.copy()
            for pt1, pt2 in zip(img_coords, img_coords[1:]):
                cv2.line(path_frame, pt1, pt2,
                         color=(0, 255, 0),
                         thickness=1,
                         lineType=cv2.LINE_AA,
                         shift=shift_bits)

            # Display the frame with the path superimposed
            cv2.imshow(" Maze Path", path_frame)
            cv2.waitKey(1)

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

            # Create the coordinates array, each row is a point in the maze frame
            maze_coords = np.zeros((img_size ** 2, 3))
            maze_coords[:, 0] = np.tile(x_coords, img_size)  # X-coord
            maze_coords[:, 1] = np.repeat(y_coords, img_size)  # Y-coord
            maze_coords[:, 2] = self.ball_pos[2]  # Z-coord

            # Get image coordinates corresponding to each of the maze frame coordinates
            img_coords = self.plate_pose.maze_to_image_coords(maze_coords)

            # Map these coordinates in the frame into a new image
            img_coords = img_coords.reshape(64, 64, 2).astype(np.float32)
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
                2d position of the ball in the image frame in (u, v) convention.
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

        # Make sure all corners were found, otherwise we can't use this frame
        if all(self.detector_fixed_points.corners_found):
            self.camera_localization_data.append(fix_pts)

        # If we have enough localization data (3 seconds @ 60fps)...
        if len(self.camera_localization_data) >= 180:
            # Average all the fixed marker positions we collected to get a robust measurement
            all_data = np.stack(self.camera_localization_data)
            fix_pts = all_data.mean(axis=0)

            # Let the inner corner markers Detector object know where these are as well, so it can hide them as necessary
            self.detector.fixed_corners = fix_pts

            # Estimate the pose of camera wrt the world frame: T__W_C
            self.plate_pose.camera_localization(fix_pts)

            # The camera is now localized
            self.camera_localized = True

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
        coords = coords.astype(float)[:, [1, 0]]   # We need these as (u, v) floating-point values

        # Get (normalized) 3D camera frame coordinates for every pixel
        camera_points = self.plate_pose.image_to_camera_coords(coords)

        # Define the known depth of these points
        camera_points *= self.plate_pose.T__W_C[2, -1]   # Z-coordinate of the camera in the world frame

        # Transform these camera coordinates to world coordinates
        camera_points_h = np.column_stack((camera_points, np.ones(len(camera_points))))
        world_points = (self.plate_pose.T__W_C @ camera_points_h.T).T

        # Determine which world points are within the bounds of the game
        mask = ((world_points[:, 0] >= -self.plate_pose.MARKER_DIAM)
              & (world_points[:, 0] <= self.plate_pose.L_EXT_INT_X + self.plate_pose.MARKER_DIAM)
              & (world_points[:, 1] >= -self.plate_pose.MARKER_DIAM)
              & (world_points[:, 1] <= self.plate_pose.L_EXT_INT_Y + self.plate_pose.MARKER_DIAM))

        # Create a mask for these points, with value 255 for all pixels to keep
        self.mask = 255 * mask.reshape(h, w, 1).astype(np.uint8)

    def ball_pos_backproject(self, ball_img_coords):
        """
        Compute the 3D position of the ball in the maze frame {m}.

        Args:
            ball_img_coords: np.ndarray, dim: (2,)
                The pixel coordinates of the ball in (u, v) convention

        Returns:
            x_M: np.ndarray, dim: (3,)
                 3d position of the ball in the maze frame {m}.
                 Note: the z-coordinate of the ball in maze frame is fixed and known
                 by assumption of constant contact with the maze: z__m_b = ball_radius.
        """
        # If the ball was not located in the image, return NaN's (unknown)
        if np.any(np.isnan(ball_img_coords)):
            return np.array([np.nan, np.nan, np.nan])

        # Get the (normalized) camera coordinates for the location of the ball
        xc, yc, _ = self.plate_pose.image_to_camera_coords(ball_img_coords)
        d = PlatePoseEstimator.BALL_RAD

        # EXPLANATION:
        # We want to find the ball's position in maze coordinates, given its camera coordinates.
        # A good explanation of the math used here is available at:
        # https://docs.opencv.org/4.x/d9/d0c/group__calib3d.html
        # NOTES: Our 'K' matrix is referred to as A in the above description.

        # Let H be the transform matrix that converts coordinates in the Maze frame to coordinates in the Camera frame.
        # We need to solve the equation:
        #
        #     H Pm = s Pc      where Pm is the ball's coordinates in the maze frame: [Xm, Ym, Zm, 1]
        #                            Pc is the ball's coordinates in the camera frame: [Xc, Yc, 1]
        #                        and s is a scaling factor that defines the depth of the point in the camera frame (Zc)

        # However, we impose an additional constraint on the equations - we know the Z-coordinate of Pm,
        # which is fixed to the ball's radius (d), since we assume that the ball is in contact with the maze.
        # Thus, we want to fix Zm, and solve for Xm and Ym only.

        # Thus, if we define Pm as [Xm, Ym, d, 1] (a homogeneous vector), we have the equation:
        #
        #         H Pm = s Pc
        #
        # This system has 3 equations and 3 unknowns (Xm, Ym, and s).
        # To solve it, we can first use the last equation implied by this system to solve for s.
        # The last equation implied by this system is:
        #
        #         h_31*Xm + h_32*Ym + h33_d + h34 = s
        #
        # We can now substitute this expression for s on the right-hand side of the first two equations to eliminate s,
        # and solve the remaining two equations for the other unknowns: Xm and Ym
        # Multiplying out the expressions and gathering the terms for Xm and Ym on the left-hand side
        # (and moving all other terms only involving H, d, Xc, and Yc to the right-hand side),
        # we end up with a set of two equations in two unknowns (Xm and Ym) which we can solve using NumPy.

        # Define H = first 3 rows of T__C_M, and get vars for its individual elements
        H = self.plate_pose.T__C_M[:3, :]
        h_11, h_12, h_13, h_14 = H[0, :]
        h_21, h_22, h_23, h_24 = H[1, :]
        h_31, h_32, h_33, h_34 = H[2, :]

        # Form the 2x2 matrix A implied by the final two equations as described above:
        # NOTE: this only uses terms from the first-two columns of H, as expected
        A = np.array(
            [[h_11 - h_31 * xc, h_12 - h_32 * xc],
             [h_21 - h_31 * yc, h_22 - h_32 * yc]]
        )

        # Form the right-hand side of the system as described above
        # NOTE: the right-hand side only includes known terms: H, d, Xc, and Yc, as expected
        b = np.array(
            [
                h_33 * d * xc + h_34 * xc - h_13 * d - h_14,
                h_33 * d * yc + h_34 * yc - h_23 * d - h_24
            ]
        )

        # Solve this system for the 2 unknowns, which represent the ball's coordinates in the maze frame: Xm and Ym
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
