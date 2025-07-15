import numpy as np
from typing import Tuple
from math import cos, sin, tan, atan, pi


class FiniteDiff:
    """
    State estimation using finite difference
    """

    def __init__(self, fps: float, FiniteDiff_mean_steps: int) -> None:
        self.fps = fps
        self.T_s = 1.0 / self.fps
        # self.prev_measurement = None
        self.mean_steps = FiniteDiff_mean_steps
        self.prev_measurements = np.zeros((self.mean_steps, 2))

    def initialize(self) -> Tuple[np.ndarray, np.ndarray]:
        x_0 = np.array([0.014, 0.104, 0, 0])
        P_0 = np.eye(4)

        return x_0, P_0

    def estimate(
        self,
        inputs: np.ndarray,
        measurement: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        # if self.prev_measurement is not None:
        #     finite_diff = (measurement - self.prev_measurement) / self.T_s
        # else:
        #     finite_diff = np.zeros((2,))
        prev_mean = self.prev_measurements.mean(axis=0)
        self.prev_measurements = np.concatenate(
            (self.prev_measurements[-self.mean_steps + 1 :], measurement.reshape(1, 2)),
            axis=0,
        )
        curr_mean = self.prev_measurements.mean(axis=0)
        finite_diff = (curr_mean - prev_mean) / self.T_s

        x = np.hstack((measurement, finite_diff))

        self.prev_measurement = measurement

        return x, np.eye(4)


class KF:
    """
    Kalman Filter class
    This Kalman Filter is used to estimate the state of the system, incorporating the measurements we take
    from the camera images.

    State vector = [xb, yb, xb_dot, yb_dot]   (dots are velocities, i.e., derivatives)
    The state vector at time k is denoted: x_k
    """

    def __init__(self, fps: float):
        self.g = 9.81
        self.fps = fps
        self.T_s = 1 / self.fps

        # Set up the matrices we'll need for the Kalman filter:

        # DT linear model dynamics (from basic physics)
        # A is the "Transition Model" that describes the idealized next state of the system.
        # It is used as: x_k+1 = Ax_k
        # Expanded, we have (assuming constant velocity):
        # xb_k+1 = xb_k + T_s*xb_dot_k
        # yb_k+1 = yb_k + T_s*yb_dot_k
        # xb_dot_k+1 = xb_dot_k
        # yb_dot_k+1 = yb_dot_k
        self.A = np.array(
            [[1, 0, self.T_s, 0],
             [0, 1, 0, self.T_s],
             [0, 0, 1, 0],
             [0, 0, 0, 1]]
        )

        # Matrix B encodes how the state will be impacted by the inputs to the system (u),
        # which are the (sines of the) angles of the board.
        # The inputs are incorporated during the "prior update" step as: Bu

        # For the ball's position, B encodes the final term of the position under acceleration formula:
        # x = x₀ + v₀t + (1/2)at²
        # The velocities will just be increased by: a*t

        # When a solid, uniform ball rolls down an inclined plane without slipping,
        # its acceleration (a) is given by: a = (5/7) * g * sin(θ), where 'θ' is the angle of the incline
        self.B = np.array(
            [
                [0.5 * (5/7) * self.g * self.T_s ** 2, 0],
                [0, 0.5 * (5/7) * self.g * self.T_s ** 2],
                [(5/7) * self.g * self.T_s, 0],
                [0, (5/7) * self.g * self.T_s]
            ]
        )

        # Q is the covariance of the process noise: v ~ N(0, Q),  Q: [4x4]
        # self.sigma_a = 0.1
        # self.Q = np.diag([0, 0,  self.sigma_v**2,  self.sigma_v**2]) # $$ to change ...
        self.sigma_angle = np.deg2rad(10)   # Noise in the angles measurements (here treated as inputs)
                                            # Again, is the input really sin(angle)???
        self.Q = (
            self.B @ np.diag([self.sigma_angle ** 2, self.sigma_angle ** 2]) @ self.B.T
        )

        # DT linear measurement model
        # H is the "Observation Model", which relates the state vector to the measurements vector
        # Here, we only measure position, not velocity, so only the first two terms of the state vector are measured
        self.H = np.array([[1, 0, 0, 0],
                           [0, 1, 0, 0]])

        # R is the covariance of the measurement (observation) noise: w ~ N(0, R), R: [2x2]
        self.sigma_m = 1e-4    # Assumed error in our position measurements
        self.R = np.diag([self.sigma_m ** 2, self.sigma_m ** 2])

        # Determine an initial state estimation
        self.xm, self.Pm = self.initialize()

    def initialize(
        self,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Initialize the estimator with the mean and covariance of the initial
        estimate.

        Returns:
            xm : np.ndarray, dim: (4,)
                The mean of the initial state estimate. The order of states is
                given by x = [xb, yb, xb_dot, yb_dot].
            Pm : np.ndarray, dim: (4, 4)
                The covariance of the initial state estimate. The order of
                states is given by x = [xb, yb, xb_dot, yb_dot].
        """
        # Our initial state
        # The ball is assumed to be at (xb0, yb0) with zero velocity
        xb0 = 0.014     # UPDATE THIS!
        yb0 = 0.104     # UPDATE THIS!
        x_0 = np.array([xb0, yb0, 0, 0])

        # Our initial error (covariance matrix)
        sigma_p0 = 1
        sigma_v0 = 1
        P_0 = np.diag([sigma_p0 ** 2, sigma_p0 ** 2, sigma_v0 ** 2, sigma_v0 ** 2])

        # Return our initial state
        return x_0, P_0

    def estimate(
        self,
        inputs: np.ndarray,
        measurement: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply the Kalman Filter to estimate the next state of the system given some inputs and measurements.

        Args:
            inputs : np.ndarray, dim: (2,)
                System inputs from time step k-1, u_k-1.
                The order of the inputs is given by u = [alpha, beta]   # OR IS IT (beta, -alpha) ???
                These are the board angles
            measurement : np.ndarray, dim: (2,)
                Sensor measurements from time step k, z(k).
                The order of the measurements is given by z = [xb, yb]
                This the measured position of the ball

        Returns:
            xm : np.ndarray, dim: (4,)
                The mean of the posterior estimate xm_k.
                The order of states is given by x = [xb, yb, xb_dot, yb_dot]
            Pm : np.ndarray, dim: (4, 4)
                The covariance of the posterior estimate Pm(k).
                The order of states is given by x = [xb, yb, xb_dot, yb_dot]
        """
        # The Kalman Filter is a two-step process:

        # Prior update = apply the transition model
        xp, Pp = self.prior_update(self.xm, self.Pm, inputs)

        # Measurement update = incorporate the measurements to refine the state estimation
        self.xm, self.Pm = self.measurement_update(xp, Pp, measurement)

        # Return our updated (mean) state estimate (xm) and state covariance matrix (Pm)
        return self.xm, self.Pm

    def prior_update(self, xm_prev: np.ndarray, Pm_prev: np.ndarray, u: np.ndarray):

        # WHERE's the sin(theta) of the angles???
        # Why are we multiplying by the angles themselves (u)???
        # SHOULD THIS BE self.B @ np.sin(u)???
        xp = self.A @ xm_prev - self.B @ u     # According to our conventions, positive angles will *decelerate* the ball in that direction
        Pp = self.A @ Pm_prev @ self.A.T + self.Q

        return xp, Pp

    def measurement_update(
        self,
        xp: np.ndarray,
        Pp: np.ndarray,
        measurement: np.ndarray
    ):
        # If we don't have a valid measurement, the best we can do is use the result of the transition model ("prior update")
        if np.any(np.isnan(measurement)):
            return xp, Pp

        # Kalman gain (K)
        S = self.H @ Pp @ self.H.T + self.R
        S_inv = np.linalg.inv(S)
        K = Pp @ self.H.T @ S_inv

        # Mean update
        y_tilde = measurement - self.H @ xp   # The "innovation", or measurement residual
        xm = xp + K @ y_tilde

        # Variance update
        I = np.eye(xp.shape[0])
        I_min_KH = I - K @ self.H
        Pm = I_min_KH @ Pp @ I_min_KH.T + K @ self.R @ K.T
        # NOTE: If using the "optimal Kalman gain", this simplifies to:
        # Pm = I_min_KH @ Pp

        # Return our new estimates
        return xm, Pm


class KFBias:
    """
    Kalman Filter class
    """

    def __init__(self, fps: float):
        self.g = 9.81
        self.fps = fps
        self.T_s = 1 / self.fps

        T_s = self.T_s
        g = self.g
        # DT linear model dynamics
        self.A = np.array(
            [
                [1, 0, T_s, 0, 0, (5 * T_s ** 2 * g) / 14],
                [0, 1, 0, T_s, -(5 * T_s ** 2 * g) / 14, 0],
                [0, 0, 1, 0, 0, (5 * T_s * g) / 7],
                [0, 0, 0, 1, -(5 * T_s * g) / 7, 0],
                [0, 0, 0, 0, 1, 0],
                [0, 0, 0, 0, 0, 1],
            ]
        )

        self.B = np.array(
            [
                [0, 5 / 14 * g * T_s ** 2],
                [-5 / 14 * g * T_s ** 2, 0],
                [0, 5 / 7 * g * T_s],
                [-5 / 7 * g * T_s, 0],
                [0, 0],
                [0, 0],
            ]
        )

        # process noise v ~ N(0, Q),  Q: [4x4]
        # self.sigma_a = 0.1
        # self.Q = np.diag([0, 0,  self.sigma_v**2,  self.sigma_v**2]) # $$ to change ...
        self.sigma_angle = (
            np.pi / 180 * 1
        )  # noise on the angles measurements (here treated as inputs) (alpha, beta)
        self.Q = (
            self.B @ np.diag([self.sigma_angle ** 2, self.sigma_angle ** 2]) @ self.B.T
        )

        # DT linear measurement model
        self.H = np.array([[1, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0]])

        # measurement noise w ~ N(0, R), R: [2x2]
        self.sigma_m = 1e-4
        self.R = np.diag([self.sigma_m ** 2, self.sigma_m ** 2])

        self.xm, self.Pm = self.initialize()

    def initialize(
        self,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Initialize the estimator with the mean and covariance of the initial
        estimate.

        Returns:
            xm : np.ndarray, dim: (4,)
                The mean of the initial state estimate. The order of states is
                given by x = [xb, yb, xb_dot, yb_dot].
            Pm : np.ndarray, dim: (4, 4)
                The covariance of the initial state estimate. The order of
                states is given by x = xb, yb, xb_dot, yb_dot].
        """
        self.sigma_p0 = 1
        self.sigma_v0 = 1
        self.sigma_bias0 = 0.1
        x_0 = np.array([0.014, 0.104, 0, 0, 0, 0])
        P_0 = np.diag(
            [
                self.sigma_p0 ** 2,
                self.sigma_p0 ** 2,
                self.sigma_v0 ** 2,
                self.sigma_v0 ** 2,
                self.sigma_bias0 ** 2,
                self.sigma_bias0 ** 2,
            ]
        )
        return x_0, P_0

    def estimate(
        self,
        inputs: np.ndarray,
        measurement: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Estimate the state of the vehicle.

        Args:
            inputs : np.ndarray, dim: (4,)
                System inputs from time step k-1, u(k-1). The order of the
                inputs is given by u = [alpha, beta].
            measurement : np.ndarray, dim: (2,)
                Sensor measurements from time step k, z(k). The order of the
                measurements is given by z = [xb, yb].

        Returns:
            xm : np.ndarray, dim: (4,)
                The mean of the posterior estimate xm(k). The order of states is
                given by x = [xb, yb, xb_dot, yb_dot].
            Pm : np.ndarray, dim: (4, 4)
                The covariance of the posterior estimate Pm(k). The order of
                states is given by x = [xb, yb, xb_dot, yb_dot].
        """
        # prior update
        xp, Pp = self.prior_update(self.xm, self.Pm, inputs)

        # measurement update
        self.xm, self.Pm = self.measurement_update(xp, Pp, measurement)

        return self.xm, self.Pm

    def prior_update(self, xm_prev: np.ndarray, Pm_prev: np.ndarray, u: np.ndarray):

        xp = self.A @ xm_prev + self.B @ u
        Pp = self.A @ Pm_prev @ self.A.T + self.Q
        return xp, Pp

    def measurement_update(
        self,
        xp: np.ndarray,
        Pp: np.ndarray,
        measurement: np.ndarray,
    ):

        if np.any(np.isnan(measurement)):
            # raise Exception("There is a measurement that is NaN.")
            return xp, Pp

        # Kalman gain
        K = Pp @ self.H.T @ np.linalg.inv(self.H @ Pp @ self.H.T + self.R)
        # mean update
        z_pred = self.H @ xp
        xm = xp + K @ (measurement - z_pred)
        # variance update
        Pm = (np.eye(xp.shape[0]) - K @ self.H) @ Pp @ (
            np.eye(xp.shape[0]) - K @ self.H
        ).T + K @ self.R @ K.T
        return xm, Pm


class KFVelocityControl:
    """
    Kalman Filter class
    """

    def __init__(self, fps: float):
        self.g = 9.81
        self.fps = fps
        self.T_s = 1 / self.fps

        # DT linear model dynamics
        self.A = np.array(
            [
                [1, 0, self.T_s, 0, 0, (5 * self.T_s ^ 2 * self.g) / 14],
                [0, 1, 0, self.T_s, -(5 * self.T_s ^ 2 * self.g) / 14, 0],
                [0, 0, 1, 0, 0, (5 * self.T_s * self.g) / 7],
                [0, 0, 0, 1, -(5 * self.T_s * self.g) / 7, 0],
                [0, 0, 0, 0, 1, 0],
                [0, 0, 0, 0, 0, 1],
            ]
        )

        self.B = np.array(
            [
                [0, 5 / 42 * self.T_s ^ 3 * self.g],
                [-5 / 42 * self.T_s ^ 3 * self.g, 0],
                [0, 5 / 14 * self.T_s ^ 2 * self.g],
                [-5 / 14 * self.T_s ^ 2 * self.g, 0],
                [self.T_s, 0],
                [0, self.T_s],
            ]
        )

        # process noise v ~ N(0, Q),  Q: [6x6]
        self.sigma_angle_dot = np.pi / 180 * 1
        self.Q = (
            self.B @ np.diag([self.sigma_angle ** 2, self.sigma_angle ** 2]) @ self.B.T
        )

        # DT linear measurement model
        self.H = np.array(
            [
                [1, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0],
                [0, 0, 0, 0, 1, 0],
                [0, 0, 0, 0, 0, 1],
            ]
        )

        # measurement noise w ~ N(0, R), R: [4x4]
        self.sigma_p = 1e-3  # noise on the position measurement (x,y)
        self.sigma_angle = (
            np.pi / 180 * 0.5
        )  # noise on the angles measurements (alpha, beta)
        self.R = np.diag(
            [
                self.sigma_p ** 2,
                self.sigma_p ** 2,
                self.sigma_angle ** 2,
                self.sigma_angle ** 2,
            ]
        )

        self.xm, self.Pm = self.initialize()

    def initialize(
        self,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Initialize the estimator with the mean and covariance of the initial
        estimate.

        Returns:
            xm : np.ndarray, dim: (6,)
                The mean of the initial state estimate. The order of states is
                given by x = [xb, yb, xb_dot, yb_dot, alpha, beta].
            Pm : np.ndarray, dim: (6, 6)
                The covariance of the initial state estimate. The order of
                states is given by x = [xb, yb, xb_dot, yb_dot, alpha, beta].
        """
        self.sigma_p0 = 1
        self.sigma_v0 = 1
        self.sigma_angles0 = np.pi / 180 * 10
        x_0 = np.array([0.014, 0.104, 0, 0, 0, 0])
        P_0 = np.diag(
            [
                self.sigma_p0 ** 2,
                self.sigma_p0 ** 2,
                self.sigma_v0 ** 2,
                self.sigma_v0 ** 2,
                self.sigma_angles0 ** 2,
                self.sigma_angles0 ** 2,
            ]
        )
        return x_0, P_0

    def estimate(
        self,
        inputs: np.ndarray,
        measurement: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Estimate the state of the vehicle.

        Args:
            inputs : np.ndarray, dim: (2,)
                System inputs from time step k-1, u(k-1). The order of the
                inputs is given by u = [alpha_dot, beta_dot].
            measurement : np.ndarray, dim: (4,)
                Sensor measurements from time step k, z(k). The order of the
                measurements is given by z = [xb, yb, alpha, beta].

        Returns:
            xm : np.ndarray, dim: (6,)
                The mean of the posterior estimate xm(k). The order of states is
                given by x = [xb, yb, xb_dot, yb_dot, alpha, beta].
            Pm : np.ndarray, dim: (6, 6)
                The covariance of the posterior estimate Pm(k). The order of
                states is given by x = [xb, yb, xb_dot, yb_dot, alpha, beta].
        """
        # prior update
        xp, Pp = self.prior_update(self.xm, self.Pm, inputs)

        # measurement update
        self.xm, self.Pm = self.measurement_update(xp, Pp, measurement)

        return self.xm, self.Pm

    def prior_update(self, xm_prev: np.ndarray, Pm_prev: np.ndarray, u: np.ndarray):

        xp = self.A @ xm_prev + self.B @ u
        Pp = self.A @ Pm_prev @ self.A.T + self.Q
        return xp, Pp

    def measurement_update(
        self,
        xp: np.ndarray,
        Pp: np.ndarray,
        measurement: np.ndarray,
    ):

        if np.any(np.isnan(measurement)):
            raise Exception("There is a measurement that is NaN.")

        # Kalman gain
        K = Pp @ self.H.T @ np.linalg.inv(self.H @ Pp @ self.H.T + self.R)
        # mean update
        z_pred = self.H @ xp
        xm = xp + K @ (measurement - z_pred)
        # variance update
        Pm = (np.eye(xp.shape[0]) - K @ self.H) @ Pp @ (
            np.eye(xp.shape[0]) - K @ self.H
        ).T + K @ self.R @ K.T
        return xm, Pm
