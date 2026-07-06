"""
This script dynamically generates the assets needed by the cyberrunner.xml MJCF file
"""
# Imports
from pathlib import Path
from cyberrunner_layout import cyberrunner_hard_layout
from PIL import Image, ImageDraw

# We will create several files in the 'assets' subdirectory
assets_path = Path("assets")
assets_path.mkdir(parents=True, exist_ok=True)

# ----------------------------------------------------------------------------------------------------------------------
# Walls
# Walls are modeled using 4 geoms:
#  1. A rectangular box for the sides of the wall
#  2. Two vertical cylinders at the ends of the wall to make the ends rounded
#  4. A capsule along the top of the wall to make the top rounded
wall_width = .5  # in cm
wall_height = .65   # Total height of the wall in cm

# MuJoCo uses "half-width" measurements
wall_half_width = wall_width / 2   # Also the radius of the wall caps and tops
wall_box_z_coord = wall_height - wall_width    # The z-coordinate of the center of the wall box
wall_box_height = wall_box_z_coord + wall_half_width    # The vertical height of the wall box above the board surface

wall_geoms = []

# Horizontal Walls
for left_x, right_x, y, left_end, right_end in cyberrunner_hard_layout["walls_h"]:
    # Convert to centimeters
    left_x *= 100
    right_x *= 100
    y *= 100

    # Adjust the start and end positions based on the wall ends
    # The horizontal walls were measured to "butt up" against the vertical walls
    # Horizontal walls that get a rounded "end cap" need to be shortened since the cap adds length
    # Walls that intersect another wall at a corner need to be lengthened to model the corner properly
    # I've added two fields to the wall data in cyberrunner_layout to indicate whether each end of the horizontal walls
    # needed to be shortened (-1) or lengthened (+1)
    left_x -= left_end * wall_half_width
    right_x += right_end * wall_half_width

    wall_geoms.append(f'<geom type="box" size="{wall_half_width:.3f}" fromto="{left_x:.3f} {y:.3f} {wall_box_z_coord:.3f} {right_x:.3f} {y:.3f} {wall_box_z_coord:.3f}" />')
    wall_geoms.append(f'<geom type="cylinder" size="{wall_half_width:.3f}" fromto="{left_x:.3f} {y:.3f} 0 {left_x:.3f} {y:.3f} {wall_box_height:.3f}" />')
    wall_geoms.append(f'<geom type="cylinder" size="{wall_half_width:.3f}" fromto="{right_x:.3f} {y:.3f} 0 {right_x:.3f} {y:.3f} {wall_box_height:.3f}" />')
    wall_geoms.append(f'<geom type="capsule" size="{wall_half_width:.3f}" fromto="{left_x:.3f} {y:.3f} {wall_box_height:.3f} {right_x:.3f} {y:.3f} {wall_box_height:.3f}" />')

# Vertical Walls
for bottom_y, top_y, x in cyberrunner_hard_layout["walls_v"]:
    # Convert to centimeters
    bottom_y *= 100
    top_y *= 100
    x *= 100

    # Make room for the rounded end caps,
    # but not if the wall butts up against the edge of the board
    if bottom_y > 0:
        bottom_y += wall_half_width
    if top_y < 23.1:
        top_y -= wall_half_width

    wall_geoms.append(f'<geom type="box" size="{wall_half_width:.3f}" fromto="{x:.3f} {bottom_y:.3f} {wall_box_z_coord:.3f} {x:.3f} {top_y:.3f} {wall_box_z_coord:.3f}" />')
    wall_geoms.append(f'<geom type="cylinder" size="{wall_half_width:.3f}" fromto="{x:.3f} {bottom_y:.3f} 0 {x:.3f} {bottom_y:.3f} {wall_box_height:.3f}" />')
    wall_geoms.append(f'<geom type="cylinder" size="{wall_half_width:.3f}" fromto="{x:.3f} {top_y:.3f} 0 {x:.3f} {top_y:.3f} {wall_box_height:.3f}" />')
    wall_geoms.append(f'<geom type="capsule" size="{wall_half_width:.3f}" fromto="{x:.3f} {bottom_y:.3f} {wall_box_height:.3f} {x:.3f} {top_y:.3f} {wall_box_height:.3f}" />')

walls_xml = f"""<mujoco model="cyberrunner_walls">
    {"\n    ".join(wall_geoms)}
</mujoco>
"""

# Write out the final XML
Path("cyberrunner_walls.xml").write_text(walls_xml)

# ----------------------------------------------------------------------------------------------------------------------
# Board texture image
# Supersample the image by 100 (in mm)
img = Image.new("RGB", (276 * 100, 231 * 100), color=(210, 180, 140))
draw = ImageDraw.Draw(img)

# Draws the holes
# NOTE: You may want to draw the holes on the labyrinth.png file to line up the mesh geometry with the texture map image
# using Blender. Once the UV mapping is created and the new .obj file is exported from Blender, you should then recreate
# the labyrinth.png file WITHOUT the holes drawn so that the mesh geometry alone defines the hole locations.
draw_holes = False   # Set to True if re-mapping the UV coordinates in Blender
if draw_holes:
    hole_radius = 7  # (mm)
    for x, y in cyberrunner_hard_layout["holes"]:
        # Convert to millimeters * 100
        x *= 1000 * 100
        y *= 1000 * 100

        # Flip the y-axis
        y = 23100 - y

        draw.circle((x, y), hole_radius * 100, fill="black")

# Collect the maze waypoints
points = []
for x, y in cyberrunner_hard_layout["waypoints"]:
    # Convert to millimeters * 100
    x *= 1000 * 100
    y *= 1000 * 100

    # Flip the y-axis
    y = 23100 - y

    points.append((x, y))

# Draw the line segments
draw.line(points, fill=(94, 38, 12), width=200, joint="curve")

# Resize the image to 10X the final size for higher quality
# MuJoCo will scale the texture image to the correct size
img = img.resize((276 * 10, 231 * 10), Image.Resampling.LANCZOS)

# Save the result
img.save(assets_path / "labyrinth.png")

# ----------------------------------------------------------------------------------------------------------------------
# Outer Frame Markers
# Supersample the image by 100 (in mm)
img = Image.new("RGB", (270 * 100, 270 * 100), color=(210, 180, 140))
draw = ImageDraw.Draw(img)

# Draws the blue markers
half_width = 6.2
half_height = 3.4
x_center = 135
y_center = 209
draw.ellipse(((x_center - half_width) * 100, (y_center - half_height) * 100, (x_center + half_width) * 100, (y_center + half_height) * 100), fill="blue")

# Resize the image to 10X the final size for higher quality
# MuJoCo will scale the texture image to the correct size
img = img.resize((270 * 10, 270 * 10), Image.Resampling.LANCZOS)

# Save the result
img.save(assets_path / "outer_markers_up.png")

img = Image.new("RGB", (270 * 100, 270 * 100), color=(210, 180, 140))
draw = ImageDraw.Draw(img)
y_center = 61
draw.ellipse(((x_center - half_width) * 100, (y_center - half_height) * 100, (x_center + half_width) * 100, (y_center + half_height) * 100), fill="blue")
img = img.resize((270 * 10, 270 * 10), Image.Resampling.LANCZOS)
img.save(assets_path / "outer_markers_down.png")

# ----------------------------------------------------------------------------------------------------------------------
# Inner Frame Markers
# Supersample the image by 100 (in mm)
img = Image.new("RGB", (100 * 100, 100 * 100), color=(210, 180, 140))
draw = ImageDraw.Draw(img)

# Draws the blue markers
draw.circle((50 * 100, 50 * 100), 49 * 100, fill='blue')

# Resize the image to 10X the final size for higher quality
# MuJoCo will scale the texture image to the correct size
img = img.resize((100 * 10, 100 * 10), Image.Resampling.LANCZOS)

# Save the result
img.save(assets_path / "inner_markers_front.png")

# ----------------------------------------------------------------------------------------------------------------------
