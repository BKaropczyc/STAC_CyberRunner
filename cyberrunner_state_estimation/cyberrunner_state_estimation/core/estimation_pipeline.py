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

    def estimate(self, frame, return_ball_subimg=False):
        """
        Compute the measurements and estimate the state from frame.

        Args:
            frame: np.ndarray, an image from the camera, dim: (400, 640, 3)
            return_ball_subimg: bool

        Returns:
            x_hat: np.ndarray, dim: (n_states,)
            P: np.ndarray dim: (n_states, n_states)
                covariance matrix
            inputs: np.ndarray dim: (2,)
                [alpha, beta]
            ball_subimg: optional, np.ndarray, dim: (64, 64, 3)
        """
        t0 = time.time()
        self.measurements.process_frame(frame, return_ball_subimg)
        xb, yb, _ = self.measurements.get_ball_position_in_maze()
        if return_ball_subimg:
            ball_subimg = self.measurements.get_ball_subimg()
        inputs = self.measurements.get_plate_pose()  # alpha, beta
        tmeas = time.time() - t0

        t0 = time.time()
        x_hat, P = self.estimator.estimate(
            inputs=inputs, measurement=np.array([xb, yb])
        )

        if type(self.estimator).__name__ == "KFBias":
            alpha_est = inputs[0] + x_hat[4]
            beta_est = inputs[1] + x_hat[5]
        else:  # type(self.estimator).__name__ == "KF":
            alpha_est = inputs[0]
            beta_est = inputs[1]
        alpha_est *= 180 / np.pi
        beta_est *= 180 / np.pi
        testimator = time.time() - t0

        # np.set_printoptions(precision=3)
        np.set_printoptions(formatter={"float": "{: 0.3f}".format}, precision=3)

        if self.print_measurements:
            print(
                f"ball: ({xb:6.3f}, {yb:>6.3f}) | (a, b): ({inputs[0]*180/np.pi:>5.2f}, {inputs[1]*180/np.pi:>5.2f}) [deg] | tmeas:{1000*tmeas:5.2f} [ms] | x_hat:{x_hat} | ab_est:({alpha_est:5.2f}, {beta_est:5.2f}) [deg]"
            )

        if return_ball_subimg:
            return x_hat, P, inputs, ball_subimg, xb, yb
        else:
            return x_hat, P, inputs, xb, yb
