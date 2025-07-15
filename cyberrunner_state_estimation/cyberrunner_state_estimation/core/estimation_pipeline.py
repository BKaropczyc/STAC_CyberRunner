import numpy as np
import time
import cv2
import os
from cyberrunner_state_estimation.core.measurements import Measurements
from cyberrunner_state_estimation.core.estimator import KF, KFBias, FiniteDiff
from ament_index_python.packages import get_package_share_directory


class EstimationPipeline:
    """A Class for estimating the physical state of the environment from an image."""
    def __init__(
        self,
        fps,                         # The assumed frame rate of the images being processed
        estimator="KF",              # The type of estimator to use
        FiniteDiff_mean_steps=0,
        print_measurements: bool = False,   # Whether to print the estimates as they are generated
        show_3d_anim=False,                 # Whether to display a 3D visualization of the estimated physical state
        viewpoint="side",                   # The view to use for the 3D visualization
        show_subimage_masks=False           # Whether the Detector object should show the subimage masks
    ):

        # Read in the markers.csv data generated during the "select_markers" calibration step
        share = get_package_share_directory("cyberrunner_state_estimation")
        markers = np.loadtxt(os.path.join(share, "markers.csv"), delimiter=",")

        # Create our Measurements object
        self.measurements = Measurements(
            markers=markers,
            show_3d_anim=show_3d_anim,
            viewpoint=viewpoint,
            show_subimage_masks=show_subimage_masks
        )

        # Create our estimator object
        if estimator == "FiniteDiff":
            self.estimator = FiniteDiff(fps, FiniteDiff_mean_steps)
        elif estimator == "KF":
            self.estimator = KF(fps)
        elif estimator == "KFBias":
            self.estimator = KFBias(fps)

        # Remember our other params
        self.print_measurements = print_measurements
        if print_measurements:
            np.set_printoptions(precision=3, floatmode="fixed", suppress=True, sign=" ", nanstr="  nan ")

    def estimate(self, frame, return_ball_subimg=False):
        """
        Compute the measurements and estimate the state from frame.

        Args:
            frame: np.ndarray, an image from the camera, dim: (400, 640, 3)
            return_ball_subimg: bool

        Returns:
            ball_pos: np.ndarray, dim: (2,)
                The (x,y) position of the ball in the maze frame
            board_angles: np.ndarray dim: (2,)
                [alpha, beta]
            x_hat: np.ndarray, dim: (n_states,)
                The predicted next state of the system
                Estimated state is: [xb, yb, xb_dot, yb_dot]
            ball_subimg: optional, np.ndarray, dim: (64, 64, 3)
                A 64x64 image around the ball in the maze frame
        """

        # Calculate measurements from the frame...
        self.measurements.process_frame(frame, return_ball_subimg)
        # ...and get the results
        ball_pos = self.measurements.get_ball_position_in_maze()[:2]   # We only care about the X and Y coords
        board_angles = self.measurements.get_plate_pose()
        if return_ball_subimg:
            ball_subimg = self.measurements.get_ball_subimg()

        # Get predictions of the next state of the system
        # x_hat is the predicted state of the system: [xb, yb, xb_dot, yb_dot]
        x_hat, _ = self.estimator.estimate(inputs=board_angles, measurement=ball_pos)

        # Print out the calculated measurements
        if self.print_measurements:
            print(f"ball_pos: {ball_pos} (m)  |  board_angles: {np.rad2deg(board_angles)} (deg)  |  x_hat: {x_hat}")

        # Return our results
        if return_ball_subimg:
            return ball_pos, board_angles, x_hat, ball_subimg
        else:
            return ball_pos, board_angles, x_hat
