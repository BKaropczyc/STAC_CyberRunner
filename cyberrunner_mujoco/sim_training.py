import mujoco
import cv2
import rclpy
from cyberrunner_camera.scripts.sim_publisher import SimPublisher


# Load the model, state of the simulation and the renderer
model = mujoco.MjModel.from_xml_path("cyberrunner_mujoco/cyberrunner.xml")
data = mujoco.MjData(model)
renderer = mujoco.Renderer(model, height=720, width=1280)

visuals = True
# Start as a ROS Node
rclpy.init()
sim_publisher = SimPublisher(display=False)

while True:
    #data.ctrl[:] = dreamerv3 induced action

    # Advance (0.018 simulated seconds)
    mujoco.mj_step(model, data, nstep=36)

    # Generate the array from the resulting state.
    renderer.update_scene(data, camera="fov_camera")
    pixels = renderer.render()

    # Send this observation through the camera ROS topic.
    sim_publisher.publish_frame(pixels)

    if visuals:
        bgr_pixels = cv2.cvtColor(pixels, cv2.COLOR_RGB2BGR)
        cv2.imshow("MuJoCo Camera View", bgr_pixels)
        cv2.pollKey()