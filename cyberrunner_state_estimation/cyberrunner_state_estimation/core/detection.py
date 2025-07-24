import numpy as np
from math import ceil
import cv2
from cyberrunner_state_estimation.core import gaussian_robust, masking


class Detector:
    """
    A class for detecting the location of the ball and the board corners within an image.
    All location coordinates are measured in pixels as (u, v) = (horizontal, vertical) convention,
    with the image origin in the upper-left corner.
    """

    # DEBUGGING OPTIONS:
    # Set SHOW_REGIONS to True to display the cropping regions overlaid on the frame
    # Set SHOW_LOCATIONS to True to display the detected locations of the corners and ball
    SHOW_REGIONS = False
    SHOW_LOCATIONS = False

    # Parameters for detecting the markers placed on the corners of the board
    CORNERS_CROP_SIZE_SMALL = 22   # The side length of the cropping square when detecting corners
    CORNERS_CROP_SIZE_LARGE = 50   # The crop size to use when the current position of a corner is unknown
    CORNERS_HSV_RANGES = (         # Masking parameters used when detecting corners
        (90, 111),   # (minHue, maxHue)
        (80, 255),   # (minSat, maxSat)
        (15, 255)    # (minVal, maxVal)
    )
    CORNERS_PERCENTILE = 5        # Gaussian detection q-th percentile
    CORNERS_THRESHOLD = 0.002     # Gaussian detection threshold

    # Parameters for detecting the ball
    BALL_CROP_SIZE = 50           # The side length of the cropping square when detecting the ball
    BALL_HSV_RANGES = (           # Masking parameters used when detecting the ball
        (89, 121),   # (minHue, maxHue)
        (155, 255),  # (minSat, maxSat)
        (15, 255)    # (minVal, maxVal)
    )
    BALL_PERCENTILE = 6           # Gaussian detection q-th percentile
    BALL_THRESHOLD = 10 ** (-4)   # Gaussian detection threshold

    def __init__(
        self,
        markers,                    # Initial locations of the 4 markers on the board corners, when level
        markers_are_static=False,   # Whether the markers are the static (outer) dots, vs. the moveable (inner) dots
        show_subimage_masks=False   # Whether to display the cropped subimage masks during detection
    ):
        # Store parameters in our instance vars
        self.markers = markers.astype(int)
        self.markers_are_static = markers_are_static
        self.show_subimage_masks = show_subimage_masks

        # Initialize to no known ball or corner positions
        self.ball_pos = None
        self.is_ball_found = False          # Whether the ball position was successfully detected in the most recent frame
        self.corners = None                 # The current positions of the 4 corner markers identified by 'markers'
        self.corners_found = [False] * 4    # Whether each of the 4 corners was successfully detected in the most recent frame
        self.fixed_corners = None           # The positions of the 4 outer markers on the board that don't move
                                            # NOTE: This is set externally by using a second Detector object

        # Set up the default coordinates for the ball subimage, when the ball was not detected
        # Since the ball could be anywhere, these default coordinates will be the entire board:
        self.full_board_coords = (self.markers[3], self.markers[1])    # (Upper Left marker, Lower Right marker)

        if Detector.SHOW_REGIONS:
            # Prepare to collect the rectangular regions we wish to draw for debugging purposes
            # Regions will be stored as tuples: (upper_left_coord, lower_right_coord, color)
            self.rects_to_draw = []

    def process_frame(self, frame):
        """
        Process frame to get raw image coordinates of the four corners and the ball.
        This is the primary method called by users of a Detector object.

        Args :
            frame: np.ndarray, dim: (400, 640, 3)
                The image to process

        Returns :
            corners: np.ndarray, dim: (4,2)
                The image coordinates of the four corner dots in (u, v) convention.
            ball: np.ndarray, dim: (2,)
                The image coordinates of the ball in (u, v) convention.
        """

        corners = self.detect_corners(frame)
        ball = self.detect_ball(frame)

        # Display debugging information, if requested
        if Detector.SHOW_REGIONS or Detector.SHOW_LOCATIONS:
            debugging_frame = frame.copy()

            if Detector.SHOW_REGIONS:
                # Draw each of the cropping regions, and reset rects_to_draw
                for (ul, lr, color) in self.rects_to_draw:
                    cv2.rectangle(debugging_frame, ul, lr, color, thickness=1)
                self.rects_to_draw.clear()

            if Detector.SHOW_LOCATIONS:
                self.draw_corners(debugging_frame)
                self.draw_ball(debugging_frame)

            # Display the debugging frame with the requested overlays
            cv2.imshow("Detector Debugging", debugging_frame)
            cv2.waitKey(1)

        return corners, ball    # Both in (row, column) conventions

    # ----------------------------------------------------------------------------------------------------------------------
    # Corner Detection
    # ----------------------------------------------------------------------------------------------------------------------

    def detect_corners(self, frame):
        """Determine the coordinates of the 4 corners of the board"""

        # Initialize all coordinates with zeros
        corners = np.zeros(shape=(4, 2), dtype="float32")
        corners_found = [False] * 4

        # Get cropped subimages and coordinates for each of the corners.
        cropped_corners_imgs, corners_imgs_coords = self.get_corner_subimages(frame)

        # From these cropped subimages, attempt to find the coordinates of each corner
        for i, sub_im in enumerate(cropped_corners_imgs):
            corners[i, :], corners_found[i] = self.detect_corner(sub_im, i, corners_imgs_coords[i][0])

        # Store and return the corner coordinates
        self.corners = corners
        self.corners_found = corners_found

        return corners

    def get_corner_subimages(self, im: np.ndarray):
        """
        Return cropped subimages and cropping coordinates for all 4 corners.
        If we know the current location of a corner, we'll use it to crop and predict the new location.
        Otherwise, we'll use a default cropping region based on the original 'marker' positions.

         Args:
             im: np.ndarray
                 The full image
         Returns:
             An array of cropped corner subimages
             and a corresponding array of corner cropping coordinates
        """

        # Prepare to collect subimages and cropping coordinates for all 4 corners
        sub_imgs = []
        sub_coords = []

        # Work on a copy of the image so that any changes don't alter the original
        im = im.copy()

        # Mask out the ball (if we know where it is) to prevent it from being mistakenly detected as a corner marker
        if self.is_ball_found:
            self.draw_object_mask(im, self.ball_pos)

        # For each corner...
        for i in range(4):
            # If we detected this corner during the previous frame...
            if self.corners_found[i]:
                # ... use that location as the center of the crop
                sub_img, ul, lr = self.get_cropped(im, self.corners[i, :], Detector.CORNERS_CROP_SIZE_SMALL)
            else:
                # Otherwise, use the original 'marker' position as the center of the crop
                crop_size = Detector.CORNERS_CROP_SIZE_SMALL if self.markers_are_static else Detector.CORNERS_CROP_SIZE_LARGE
                sub_img, ul, lr = self.get_cropped(im, self.markers[i, :], crop_size)

            # Add the results for this corner
            sub_imgs.append(sub_img)
            sub_coords.append((ul, lr))

            if Detector.SHOW_REGIONS:
                # Add the cropping rectangle to our list to draw
                self.rects_to_draw.append((ul, lr, (0, 0, 255)))

        # Return the results
        return sub_imgs, sub_coords

    def detect_corner(self, sub_im: np.ndarray, i: int, coords_ul_sub_im: np.ndarray):
        # Mask the subimage using the HSV values for the corners
        _, mask = masking.mask_hsv(sub_im, Detector.CORNERS_HSV_RANGES)

        # Try to find the center of the object in the mask
        c_local, found = gaussian_robust.detect_gaussian(
            mask, i, Detector.CORNERS_PERCENTILE, Detector.CORNERS_THRESHOLD, show_sub=self.show_subimage_masks
        )

        # Calculate the global coordinates of the detected object
        c = (coords_ul_sub_im + c_local).astype("float32")

        # If the detected corner location is too far away from the original "marker" position,
        # invalidate the result. (We likely mistook the ball for the corner marker...)
        if found:
            dist_to_orig_marker = np.linalg.norm(c - self.markers[i, :])
            if dist_to_orig_marker > 30.0:
                # Our result is invalid
                found = False

        # Return our results
        return c, found

    # ----------------------------------------------------------------------------------------------------------------------
    # Ball Detection
    # ----------------------------------------------------------------------------------------------------------------------

    def detect_ball(
        self,
        im: np.ndarray
    ):
        """
        Detect and return the position of the ball in the image.
        If the position of the ball cannot be detected, set is_ball_found to False and return NaN's
        Args:
            im: np.ndarray
                The image
        Returns:
             The position of the ball as an ndarray in (u, v) convention
             or an array of NaN's if the position ball could not be determined.
        """

        # Work on a copy of the image so that any changes don't alter the original
        im = im.copy()

        # Mask out the corner markers, to avoid them being mistakenly detected as the ball
        for i in range(4):
            # Inner (movable) markers
            if self.corners_found[i]:
                self.draw_object_mask(im, self.corners[i, :])
            else:
                self.draw_object_mask(im, self.markers[i, :])

            # Outer (static) markers
            self.draw_object_mask(im, self.fixed_corners[i, :])

        # Determine the cropping region in which we should look for the ball
        # If we detected the ball in the previous frame...
        if self.is_ball_found:
            # Get a cropped subimage based on the ball's most recent location
            ball_subimg, subimg_ul_coords, subimg_dr_coords = self.get_cropped(im, self.ball_pos, Detector.BALL_CROP_SIZE)

        else:
            # We didn't detect the ball in the previous frame
            # Get a cropped subimage based on the entire playing area
            ul, dr = self.full_board_coords
            ball_subimg, subimg_ul_coords, subimg_dr_coords = (
                im[ul[1]:dr[1], ul[0]:dr[0], :],
                ul, dr
            )

        if Detector.SHOW_REGIONS:
            # Add the cropping rectangle to our list to draw
            self.rects_to_draw.append((subimg_ul_coords, subimg_dr_coords, (0, 255, 0)))

        # Mask the subimage using the HSV values for the ball
        _, mask = masking.mask_hsv(ball_subimg, Detector.BALL_HSV_RANGES)

        # Try to find the center of the object in the mask
        c_local, self.is_ball_found = gaussian_robust.detect_gaussian(
            mask, 4, Detector.BALL_PERCENTILE, Detector.BALL_THRESHOLD, show_sub=self.show_subimage_masks
        )

        # If we didn't find the ball, return NaN's
        if not self.is_ball_found:
            return np.array([np.nan, np.nan])

        # Calculate the coordinates of the detected object,
        # store the ball's position, and return it
        c = (subimg_ul_coords + c_local).astype("float32")
        self.ball_pos = c
        return c

    # ----------------------------------------------------------------------------------------------------------------------
    # Utilities
    # ----------------------------------------------------------------------------------------------------------------------

    @staticmethod
    def get_cropped(im: np.ndarray, pos: np.ndarray, h_p: float, w_p: float = None):
        """
        Return a cropped image centered on a given position,
        and the coordinates of the top-left and bottom-right corners of the cropped image.

        Args :
            im: np.ndarray
                image
            pos: np.ndarray
                 position of the center of the subimage in (u, v) convention
            h_p: float
                 height of the subimage in pixels
            w_p: float
                 width of the subimage in pixels
                 If not passed, use the height

        Returns :
            im_cropped: np.ndarray
                the cropped image
            ul: np.ndarray, dim: (2,)
                top-left corner coordinates of the cropped image in (u, v) convention
            dr: np.ndarray, dim: (2,)
                down-right corner coordinates of the cropped image in (u, v) convention
        """
        # Get the full height and width of the original image
        h, w = im.shape[:2]

        # The cropped image must have an integral width & height
        if w_p is None:
            w_p = h_p
        h_p, w_p = round(h_p), round(w_p)

        # Assume pixel i covers all values in the half-open interval [i, i+1)
        # Thus, any non-integral positions will be located in pixel floor(position)
        pos = np.floor(pos)

        # Calculate the coordinates of the top-left and bottom-right corners of the cropped image
        # Be sure not to exceed the bounds of the original image
        ul_col = max(0, ceil(pos[0] - w_p / 2))
        ul_row = max(0, ceil(pos[1] - h_p / 2))
        dr_col = min(w, ceil(pos[0] + w_p / 2))      # 1 beyond the last column
        dr_row = min(h, ceil(pos[1] + h_p / 2))      # 1 beyond the last row

        # Crop the image
        im_cropped = im[ul_row:dr_row, ul_col:dr_col]

        # Form the cropping coordinates as (u, v) = (column, row)
        ul = np.array([ul_col, ul_row])
        dr = np.array([dr_col - 1, dr_row - 1])

        # Return the cropped image and cropping coordinates
        return im_cropped, ul, dr

    @staticmethod
    def draw_object_mask(im: np.ndarray, loc: np.ndarray):
        """
        Draw a red circle at the given location to hide an object (e.g., corner marker, ball) that could be mistaken
        for the thing we're currently trying to detect.
        """
        cv2.circle(
            im,
            np.round(loc).astype(int),
            radius=8,
            color=(0, 0, 255),    # Red in (B,G,R)
            thickness=cv2.FILLED
        )

    def draw_corners(self, frame: np.ndarray):
        """Draw an 'x' on the detected corner locations"""
        for i in range(4):
            if self.corners_found[i]:
                cv2.drawMarker(
                    frame,
                    position=np.round(self.corners[i]).astype(int),
                    color=(0, 0, 255),    # Red in (B,G,R)
                    markerType=cv2.MARKER_TILTED_CROSS,
                    markerSize=5,
                    thickness=1
                )

    def draw_ball(self, frame: np.ndarray):
        """Draw an 'x' on the detected ball location"""
        if self.is_ball_found:
            cv2.drawMarker(
                frame,
                position=np.round(self.ball_pos).astype(int),
                color=(0, 255, 0),    # Green in (B,G,R)
                markerType=cv2.MARKER_TILTED_CROSS,
                markerSize=5,
                thickness=1
            )
