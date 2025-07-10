#!/usr/bin/env python3

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

    # Define a maximum servo velocity to send
    max_vel = 50.0

    # Initialize the pygame module
    pygame.init()
    pygame.joystick.init()
    clock = pygame.time.Clock()

    # Find an available joystick
    joystick_count = pygame.joystick.get_count()
    if joystick_count == 0:
        print("No joysticks found.")
    else:
        # Assuming you want the first detected joystick
        joystick = pygame.joystick.Joystick(0)
        joystick.init()
        print(f"Opened joystick: {joystick.get_name()}")

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
    latest_v_pos = 0.0
    latest_h_pos = 0.0

    # Main loop:
    running = True
    while running:
        clock.tick(60)

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
                if axis == 1:
                    latest_v_pos = value
                elif axis == 3:
                    latest_h_pos = value

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
