#!/usr/bin/env python3

import pygame
import numpy as np
import rclpy
from rclpy.node import Node
from cyberrunner_interfaces.msg import DynamixelVel


def main(args=None):
    # Initialize rclpy
    rclpy.init(args=args)

    # Prepare to publish DynamixelVel messages to the cyberrunner_dynamixel/cmd topic
    # This is the topic that the cyberrunner_dynamixel node should be listening on
    node = Node("pantilt")
    pub = node.create_publisher(DynamixelVel, "cyberrunner_dynamixel/cmd", 10)
    msg = DynamixelVel()

    # Define a maximum servo velocity to send
    max_vel = np.array([50.0, -50.0])

    # Initialize the pygame module
    pygame.init()
    pygame.display.set_caption("CyberRunner Control")
    clock = pygame.time.Clock()

    print("\nUSAGE: Press the mouse button and move the mouse within the window to send velocity messages to the servos.\n"
          "Close the window or press any key to exit.")

    # Create a surface on screen that has the size of 640 x 480
    window_size = (640, 480)
    screen = pygame.display.set_mode(window_size)
    norm_const = np.array(window_size) / 2.0

    # Main loop:
    running = True
    while running:
        clock.tick(60)

        # If the user closes the pygame window, stop the script
        for e in pygame.event.get():
            if (e.type == pygame.QUIT) or (e.type == pygame.KEYDOWN):
                running = False
                continue

        # If the mouse-button is pressed...
        if pygame.mouse.get_pressed()[0]:
            # read the mouse position and covert each coordinate to [-1, 1]
            mouse_pos = np.array(pygame.mouse.get_pos(), dtype=np.float64)
            mouse_pos -= norm_const
            mouse_pos /= norm_const
        else:
            # Otherwise, force the position to 0, to stop the servos
            mouse_pos = np.zeros(2, dtype=np.float64)

        # Publish a velocity message to move the servos
        goal_vel = mouse_pos * max_vel
        msg.vel_1 = goal_vel[0]
        msg.vel_2 = goal_vel[1]
        pub.publish(msg)

    # Stop the servos
    msg.vel_1 = 0.0
    msg.vel_2 = 0.0
    pub.publish(msg)

    # Deactivate the Pygame library
    pygame.quit()


if __name__ == "__main__":
    main()
