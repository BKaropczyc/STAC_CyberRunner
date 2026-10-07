# Change Log


## v0.1.2

### Changed
    - Moved MuJoCo into sim_publisher.py so the ROS node owns the simulation loop.
    - Subscribed to the topic for control messages to apply the latest commands to the actuators.

## v0.1.1 (2026-9-26)

### Added
    - The initial implementation of sim_publisher.py, the ROS node that publishes simulation frames during training.
    - The node publishes to cyberrunner_camera/image which is subscribed to by the state estimation pipeline.
    - The node is started in sim_training.py, during the training loop, the object publishes the image as a ROS message

### Changed
    - Changed the cyberrruner.xml to increase the frame buffer limit to match the fov_cameras aspect ratio