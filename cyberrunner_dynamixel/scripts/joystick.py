#!/usr/bin/env python3

import os
import pygame
import rclpy
from rclpy.node import Node
from cyberrunner_interfaces.msg import DynamixelVel


def main(args=None):
    # Initialize rclpy
    rclpy.init(args=args)

    # Prepare to publish DynamixelVel messages to the cyberrunner_dynamixel/cmd topic
    # This is the topic that the cyberrunner_dynamixel node should be listening on
    node = Node("joystick")
    pub = node.create_publisher(DynamixelVel, "cyberrunner_dynamixel/cmd", 10)
    msg = DynamixelVel()

    # Parameters that impact the publishing of our ROS messages
    max_vel = 100.0     # Maximum servo velocity to send
    min_diff = 0.005    # Minimum difference in the JOYAXISMOTION values that will cause us to publish a new message
                        # Increasing this value prevents publishing many messages due to joystick jitter / noise

    # Allow pygame to receive joystick events even when the game window is not activated
    os.environ["SDL_JOYSTICK_ALLOW_BACKGROUND_EVENTS"] = "1"

    # Initialize the pygame module
    pygame.init()
    clock = pygame.time.Clock()

    # Find an available joystick
    joystick_count = pygame.joystick.get_count()
    for i in range(joystick_count):
        joystick = pygame.joystick.Joystick(i)
        joystick_name = joystick.get_name()
        if "joystick" in joystick_name.lower():
            # This looks like a joystick we can use
            joystick.init()
            print(f"Opened joystick: {joystick_name}")
            break
    else:
        # No valid joysticks were found
        print("No acceptable joysticks found.")
        exit(1)

    # Create game screen
    window_size = (640, 100)
    screen = pygame.display.set_mode(window_size)
    pygame.display.set_caption("CyberRunner Joystick Control")

    # Show the usage message
    font = pygame.font.SysFont("Arial", 20)
    text_surface = font.render("Move the joysticks to send velocity messages to the servos.", True, "white")
    screen.blit(text_surface, (20, 20))
    text_surface = font.render("Close the window or press any key to exit.", True, "white")
    screen.blit(text_surface, (20, 45))
    pygame.display.update()

    # Keep track of the latest position of each joystick axis
    latest_h_pos = 0.0
    latest_v_pos = 0.0

    # Main loop:
    running = True
    while running:
        # Limit the game to 60 fps
        clock.tick(60)

        # Only publish a new message if we received an applicable event
        pub_msg = False

        for event in pygame.event.get():
            # If the user closes the pygame window, stop the script
            if (event.type == pygame.QUIT) or (event.type == pygame.KEYDOWN):
                running = False
                continue
            elif event.type == pygame.JOYAXISMOTION:
                # Axis motion (e.g., analog sticks)
                axis = event.axis
                value = event.value

                # Update our joystick positions
                if axis == 0:
                    if abs(value - latest_h_pos) > min_diff:
                        latest_h_pos = value
                        pub_msg = True
                elif axis == 1:
                    if abs(value - latest_v_pos) > min_diff:
                        latest_v_pos = value
                        pub_msg = True

        if pub_msg:
            # Publish a velocity message to move the servos
            msg.vel_1 = latest_h_pos * max_vel
            msg.vel_2 = latest_v_pos * max_vel
            pub.publish(msg)

    # Stop the servos
    msg.vel_1 = 0.0
    msg.vel_2 = 0.0
    pub.publish(msg)

    # Deactivate the Pygame library
    pygame.quit()


if __name__ == "__main__":
    main()
