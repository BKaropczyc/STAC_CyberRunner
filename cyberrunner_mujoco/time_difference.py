import mujoco
import random
import cv2
import time

model = mujoco.MjModel.from_xml_path("cyberrunner.xml")
data = mujoco.MjData(model)
renderer = mujoco.Renderer(model, height=360, width=640)

display = True

# Keep track of how quickly we're publishing images
frame_count = 0
previous = time.time()
start_time = previous

while True:
    # Send some random actions
    x = random.randrange(-1000, 1000)
    y = random.randrange(-1000, 1000)
    data.actuator('tilt_x').ctrl = x
    data.actuator('tilt_y').ctrl = y

    # Advance the physics model 36 * 0.0005 = 0.018 sec ~= 1/55 of a second
    mujoco.mj_step(model, data, nstep=36)

    # Grab a new image from the "camera"
    renderer.update_scene(data, camera="fov_camera")
    pixels = renderer.render()
    frame_count += 1

    if display:
        bgr_pixels = cv2.cvtColor(pixels, cv2.COLOR_RGB2BGR)
        cv2.imshow("MuJoCo Camera View", bgr_pixels)
        cv2.pollKey()

    # Output some timing stats every ~2 seconds
    now = time.time()
    dur = now - previous
    if dur >= 2.0:
        # Frame rate
        fps = frame_count / dur
        print(f"MuJoCo camera frame rate: {fps:0.2f} fps")

        # Simulation speed
        sim_time = data.time
        wall_clock_time = now - start_time
        print(f"Simulation is running {sim_time / wall_clock_time:.2f} times as fast as wall-clock time")
        print()

        # Reset our count
        frame_count = 0
        previous = now