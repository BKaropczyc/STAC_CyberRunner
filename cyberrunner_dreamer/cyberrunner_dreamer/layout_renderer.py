import numpy as np
from math import ceil
import cv2


class LayoutRenderer:
    """
    A class for visualizing the layout of a CyberRunner maze, along with the current ball position.
    This class is useful for understanding the Gym environment's understanding of the RL environment,
    especially as it pertains to calculating the reward.

    NOTE: All measurements in the provided layout are assumed to be in meters, as measured from the lower-left
    corner of the board.
    """
    def __init__(
        self,
        layout: dict,            # A layout dictionary as defined in cyberrunner_layout.py
        scale: float=1000.0,     # Scale for the visualization. At a scale of 1000, 1mm in maze space = 1 pixel
        *,
        board_width=0.276,      # The width of the board (in meters) provided in layout
        board_height=0.231,     # The height of the board (in meters)
        path=None               # An optional Path object for overlaying the "off-path" regions on the visualization
    ):
        # Store init params as instance vars
        self.layout = layout
        self.scale = scale
        self.board_width = board_width
        self.board_height = board_height

        # Define colors
        self.board_color = (195, 219, 221)
        self.wall_color = (55, 67, 76)
        self.hole_color = (21, 26, 27)
        self.path_color = (3, 0, 183)
        self.ball_color = (255, 166, 0)
        self.off_path_color = (0, 0, 255)
        self.path_pos1_color = (0, 255, 0)
        self.path_posN_color = (177, 255, 177)

        # Other fixed dimensions
        self.hole_r = 0.0075          # The radius of a hole (in m)
        self.ball_r = 0.006           # The radius of the ball (in m)
        self.wall_r = 0.0025          # The width of a wall / radius of the end caps (in m)
        self.path_thickness = 0.001   # The thickness of the path line (in m)

        # Prepare to use sub-pixel precision in our OpenCV drawing routines ('shift' param)
        self.shift_bits = 4

        # Create an image of the layout
        self.image = np.zeros(shape=(ceil(self.board_height * self.scale),
                                    ceil(self.board_width * self.scale),
                                    3),
                              dtype=np.uint8)

        # Set the background color
        self.image[:, :] = np.array(self.board_color)

        # Draw the path
        waypoints = np.array(self.layout["waypoints"])
        # Flip the Y-axis, since OpenCV assume images have their origin in the top-left corner,
        # but the layout measurements are from the lower-left corner.
        waypoints[:, 1] = self.board_height - waypoints[:, 1]

        # Convert waypoints to pixels
        waypoints = self._to_pixels(waypoints)

        # Draw lines connecting each waypoint
        for pt1, pt2 in zip(waypoints, waypoints[1:]):
            cv2.line(self.image, pt1, pt2,
                     color=self.path_color,
                     thickness= round(self.path_thickness * self.scale),
                     lineType=cv2.LINE_AA,
                     shift=self.shift_bits)

        # Draw the holes
        holes = np.array(self.layout["holes"])
        holes[:, 1] = self.board_height - holes[:, 1]   # Flip Y-axis
        holes = self._to_pixels(holes)

        for pt in holes:
            cv2.circle(self.image, center=pt, radius=self._to_pixels(self.hole_r),
                       color=self.hole_color,
                       thickness=cv2.FILLED,
                       lineType=cv2.LINE_AA,
                       shift=self.shift_bits)

        # Draw the horizontal walls
        # Walls will be drawn as a rectangle with two round "end-caps" on the ends to mimic the physical board.
        # The width of the rectangle is reduced to account for the end-cap.
        # The given y-axis is assumed to represent the middle of the wall's horizontal position.
        walls_h = np.array(self.layout["walls_h"])    # Each row is: (start_x, end_x, y)
        walls_h[:, 2] = self.board_height - walls_h[:, 2]   # Flip Y-axis

        # Upper-left corner of the rectangle
        uls = walls_h[:, [0, 2]]
        uls[:, 0] += self.wall_r   # Inset for the end-cap
        uls[:, 1] -= self.wall_r   # y coordinate is the center of the wall
        uls = self._to_pixels(uls)

        # Lower-right corner of the rectangle
        lrs = walls_h[:, [1, 2]]
        lrs[:, 0] -= self.wall_r
        lrs[:, 1] += self.wall_r
        lrs = self._to_pixels(lrs)

        # Left end-cap
        lecs = walls_h[:, [0, 2]]
        lecs[:, 0] += self.wall_r
        lecs = self._to_pixels(lecs)

        # Right end-cap
        recs = walls_h[:, [1, 2]]
        recs[:, 0] -= self.wall_r
        recs = self._to_pixels(recs)

        # Draw each wall...
        for ul, lr, lec, rec in zip(uls, lrs, lecs, recs):
            # Draw the main rectangle
            cv2.rectangle(self.image, ul, lr,
                          color=self.wall_color,
                          thickness=cv2.FILLED,
                          lineType=cv2.LINE_AA,
                          shift=self.shift_bits)

            # Add the end-caps
            for c in [lec, rec]:
                cv2.circle(self.image, center=c, radius=self._to_pixels(self.wall_r),
                           color=self.wall_color,
                           thickness=cv2.FILLED,
                           lineType=cv2.LINE_AA,
                           shift=self.shift_bits)

        # Draw the vertical walls
        # This follows the same basic approach as the horizontal walls.
        walls_v = np.array(self.layout["walls_v"])       # Each row is: (start_y, end_y, x)
        walls_v[:, :2] = self.board_height - walls_v[:, :2]   # Flip Y-axis

        # Upper-left corner of the rectangle
        uls = walls_v[:, [2, 1]]
        uls[:, 0] -= self.wall_r
        uls[:, 1] += self.wall_r
        uls = self._to_pixels(uls)

        # Lower-right corner of the rectangle
        lrs = walls_v[:, [2, 0]]
        lrs[:, 0] += self.wall_r
        lrs[:, 1] -= self.wall_r
        lrs = self._to_pixels(lrs)

        # Top end-cap
        tecs = walls_v[:, [2, 1]]
        tecs[:, 1] += self.wall_r
        tecs = self._to_pixels(tecs)

        # Bottom end-cap
        becs = walls_v[:, [2, 0]]
        becs[:, 1] -= self.wall_r
        becs = self._to_pixels(becs)

        # Draw each wall...
        for ul, lr, tec, bec in zip(uls, lrs, tecs, becs):
            # Draw the main rectangle
            cv2.rectangle(self.image, ul, lr,
                          color=self.wall_color,
                          thickness=cv2.FILLED,
                          lineType=cv2.LINE_AA,
                          shift=self.shift_bits)

            # Add the end-caps
            for c in [tec, bec]:
                cv2.circle(self.image, center=c, radius=self._to_pixels(self.wall_r),
                           color=self.wall_color,
                           thickness=cv2.FILLED,
                           lineType=cv2.LINE_AA,
                           shift=self.shift_bits)

        # Overlay the "off-path" regions
        if path is not None:
            # Determine which entries in closest_idx represent an "off-path" coordinate
            # Flip the Y-axis to put the origin the lower-left corner
            offpath = path.closest_idx[::-1, :] == -1

            # Create a blending mask to overlay the off-path regions on the board
            blend_mask = offpath.astype(float) * 0.5     # 50% blended overlay
            blend_mask = np.expand_dims(blend_mask, -1).repeat(3, axis=-1)   # Blend all 3 colors BGR
            blend_mask = cv2.resize(blend_mask, dsize=(self.image.shape[1], self.image.shape[0]))

            # Create a 'source' image for the off-path regions (a constant color)
            op_source = np.zeros_like(blend_mask, dtype=np.uint8)
            op_source[:, :] = self.off_path_color

            # Construct the blended image
            blend = ((1 - blend_mask) * self.image + blend_mask * op_source).astype(np.uint8)
            self.image = blend

    def get_image(self, ball_pos=None, off_path=False, rel_path=None):
        """
        Return an image of the board layout, with the optional ball and nearest path point rendered as well
        Args:
            ball_pos: ndarray of dim (2,)
                The position of the ball on the board, in meters from the lower-left corner
            off_path: bool
                Whether the ball's position is considered off-path
            rel_path: ndarray of dim (p, 2)
                The next p points along the path toward the goal, relative to the ball's current position

        Returns: An ndarray of dim (height, width, 3) with the layout and game elements drawn on it
        """
        # Make a copy of our layout image to draw on
        frame = self.image.copy()

        # Draw the ball
        if ball_pos is not None:
            ball_pos = ball_pos.copy()    # Be sure not to modify the original
            ball_pos[1] = self.board_height - ball_pos[1]  # Flip Y-axis
            ball_coords = self._to_pixels(ball_pos)

            ball_color = self.off_path_color if off_path else self.ball_color

            cv2.circle(frame, center=ball_coords, radius=self._to_pixels(self.ball_r),
                       color=ball_color,
                       thickness=cv2.FILLED,
                       lineType=cv2.LINE_AA,
                       shift=self.shift_bits)

            # Draw the path points toward the goal
            if rel_path is not None:
                rel_path = rel_path.copy()    # Be sure not to modify the original
                rel_path[:, 1] *= -1    # Flip the Y-axis

                # Params for first path marker
                radius = self.ball_r / 2  # Draw the first marker at half the ball's radius
                color = self.path_pos1_color

                for i, v in enumerate(rel_path):
                    path_pt = self._to_pixels(ball_pos + v)   # Positions are relative to the ball

                    if i == 1:
                        # Params for subsequent path markers
                        radius = self.ball_r / 4   # Draw the remaining points at 1/4 the ball's radius
                        color = self.path_posN_color

                    cv2.circle(frame, center=path_pt, radius=self._to_pixels(radius),
                               color=color,
                               thickness=cv2.FILLED,
                               lineType=cv2.LINE_AA,
                               shift=self.shift_bits)

        # Return the rendered frame
        return frame

    def _to_pixels(self, coords):
        """
        Convert coordinate values to OpenCV pixel values by applying our scaling factor and "shift" parameter
        Args:
            coords: A scalar or ndarray of coordinates in meters, measured from the lower-left corner

        Returns:
            An integer or integer array of pixel values appropriate for passing to an OpenCV drawing routine.
            NOTE: The same 'shift' value must be passed to the drawing routine as well
        """
        if isinstance(coords, np.ndarray):
            return (coords * self.scale * 2 ** self.shift_bits).astype(int)
        else:
            return int(coords * self.scale * 2 ** self.shift_bits)
