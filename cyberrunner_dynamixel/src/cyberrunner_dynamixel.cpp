#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <unistd.h>
#include <unistd.h>
#include <string.h>
#include <cmath>

#include <chrono>
#include <thread>

#include "cyberrunner_dynamixel/error_handling.h"
#include "cyberrunner_dynamixel/dynamixel_controller.h"
#include "cyberrunner_interfaces/msg/dynamixel_vel.hpp"
#include "cyberrunner_interfaces/msg/state_estimate_sub.hpp"
#include "cyberrunner_interfaces/srv/dynamixel_reset.hpp"
#include "cyberrunner_interfaces/srv/dynamixel_test.hpp"
#include "rclcpp/rclcpp.hpp"


#define DYNAMIXEL_ID_1 (1)
#define DYNAMIXEL_ID_2 (2)

#define ERR(err) \
    do {exit(EXIT_FAILURE);} while(0)

// TODO: ERROR HANDLING!!!


// TODO clean this up
const char* port = "/dev/ttyUSB0";
rclcpp::Node::SharedPtr node;
rclcpp::CallbackGroup::SharedPtr se_callback_group;
int32_t positions[2];
double alpha, beta;   // Board angles


void set_dynamixel_speed(const cyberrunner_interfaces::msg::DynamixelVel::SharedPtr msg)
{
    int32_t dynamixel_ids[2];
    int32_t moving_speeds[2];

    // Moving speeds from message
    moving_speeds[0] = msg->vel_1;
    moving_speeds[1] = msg->vel_2;

    // Dynamixel ids
    dynamixel_ids[0] = DYNAMIXEL_ID_1;
    dynamixel_ids[1] = DYNAMIXEL_ID_2;

    dynamixel_step(2, dynamixel_ids, moving_speeds);
}


void set_board_angles(const cyberrunner_interfaces::msg::StateEstimateSub::SharedPtr msg)
{
    // Store the board angles (in degrees) from the message
    alpha = msg->state.alpha * (180.0 / M_PI);
    beta = msg->state.beta * (180.0 / M_PI);
}


void reset_dynamixel(const std::shared_ptr<cyberrunner_interfaces::srv::DynamixelReset::Request>,
          std::shared_ptr<cyberrunner_interfaces::srv::DynamixelReset::Response> response)
{
    // Dynamixel ids
    int32_t dynamixel_ids[2];
    dynamixel_ids[0] = DYNAMIXEL_ID_1;
    dynamixel_ids[1] = DYNAMIXEL_ID_2;

    // Reset params
    double goal_alpha = -1.0;
    double goal_beta = 0.0;
    double tol = 0.2;
    int reset_speed = 50;

    // Initialize the board angles to NaN to detect when we've started receiving state estimation updates
    alpha = std::nan("");
    beta = std::nan("");

    // Subscribe to the state estimation messages in a separate callback group
    // (so that we can still process them while resetting the board)
    rclcpp::SubscriptionOptions sub_options;
    sub_options.callback_group = se_callback_group;
    rclcpp::Subscription<cyberrunner_interfaces::msg::StateEstimateSub>::SharedPtr se_sub = node->create_subscription<cyberrunner_interfaces::msg::StateEstimateSub>("cyberrunner_state_estimation/estimate_subimg", 1, &set_board_angles, sub_options);

    // Wait for state estimation messages to start being received
    while (std::isnan(alpha) || std::isnan(beta)) {
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }

    // Keep track of the latest commands sent to the Dynamixels,
    // so that we only send commands when the values have changed
    int32_t last_speed_0 = 0;
    int32_t last_speed_1 = 0;

    // Attempt to reset the playing surface
    do {
        double alpha_diff = goal_alpha - alpha;
        double beta_diff = goal_beta - beta;

        int32_t speed_0;
        if (std::abs(alpha_diff) < tol)
            speed_0 = 0;
        else if (alpha_diff > 0)
            speed_0 = -reset_speed;
        else
            speed_0 = reset_speed;

        int32_t speed_1;
        if (std::abs(beta_diff) < tol)
            speed_1 = 0;
        else if (beta_diff > 0)
            speed_1 = -reset_speed;
        else
            speed_1 = reset_speed;

        // Send commands to the Dynamixels if necessary
        if ((speed_0 != last_speed_0) || (speed_1 != last_speed_1))
        {
            printf("Angles: alpha=%f, beta=%f\n", alpha, beta);
            printf("Sending speed_0 = %d, speed_1 = %d\n", speed_0, speed_1);

            int32_t moving_speeds[2];
            moving_speeds[0] = speed_0;
            moving_speeds[1] = speed_1;
            dynamixel_step(2, dynamixel_ids, moving_speeds);

            // Remember the latest speeds we've sent
            last_speed_0 = speed_0;
            last_speed_1 = speed_1;
        }

        // Process incoming state estimation events
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    } while (rclcpp::ok() && ((last_speed_0 != 0) || (last_speed_1 != 0)));

    response->success = 1;
}

// Service to test the dynamixel servos to make sure the servos are firmly connected to the shaft
void test_dynamixel(const std::shared_ptr<cyberrunner_interfaces::srv::DynamixelTest::Request>,
          std::shared_ptr<cyberrunner_interfaces::srv::DynamixelTest::Response> response)
{
    int32_t dynamixel_ids[2];
    int32_t moving_speeds[2];

    // Dynamixel ids
    dynamixel_ids[0] = DYNAMIXEL_ID_1;
    dynamixel_ids[1] = DYNAMIXEL_ID_2;
  
    // Reset motors
    int result = dynamixel_init(port, 2, dynamixel_ids, 1000000, 50, (uint32_t*)positions); 
    if (result != 0){
       response->success = 0;
       return;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    
    // Wiggle each servo back and forth several times
    for (int i=0; i<=5;i++){
        moving_speeds[0] = 100;
        moving_speeds[1] = 0;
        dynamixel_step(2, dynamixel_ids, moving_speeds);
        std::this_thread::sleep_for(std::chrono::milliseconds(250));
        
        moving_speeds[0] = -100;
        moving_speeds[1] = 0;
        dynamixel_step(2, dynamixel_ids, moving_speeds);
        std::this_thread::sleep_for(std::chrono::milliseconds(250));
    }
    moving_speeds[0] = 100;
    moving_speeds[1] = 0;
    dynamixel_step(2, dynamixel_ids, moving_speeds);
    std::this_thread::sleep_for(std::chrono::milliseconds(125));
    
    for (int i=0; i<=5;i++){
        moving_speeds[0] = 0;
        moving_speeds[1] = 100;
        dynamixel_step(2, dynamixel_ids, moving_speeds);
        std::this_thread::sleep_for(std::chrono::milliseconds(250));
        
        moving_speeds[0] = 0;
        moving_speeds[1] = -100;
        dynamixel_step(2, dynamixel_ids, moving_speeds);
        std::this_thread::sleep_for(std::chrono::milliseconds(250));
    }
    moving_speeds[0] = 0;
    moving_speeds[1] = 100;
    dynamixel_step(2, dynamixel_ids, moving_speeds);
    std::this_thread::sleep_for(std::chrono::milliseconds(125));

    // Stop the servos
    moving_speeds[0] = 0;
    moving_speeds[1] = 0;
    dynamixel_step(2, dynamixel_ids, moving_speeds);
    std::this_thread::sleep_for(std::chrono::milliseconds(50));

    response->success = 1;
}


int main(int argc, char** argv)
{
    // Init ROS
    rclcpp::init(argc, argv);

    // Get device name
    if(argc >= 2) port = argv[1];
    printf("Using port: %s. Use the command-line argument to change the port, e.g.:\n  ros2 run cyberrunner_dynamixel cyberrunner_dynamixel /dev/ttyUSB*\n", port);

    // Initialize dynamixel
    int32_t dynamixel_ids[2];
    dynamixel_ids[0] = DYNAMIXEL_ID_1;
    dynamixel_ids[1] = DYNAMIXEL_ID_2;
    dynamixel_init(port, 2, dynamixel_ids, 1000000, 256, (uint32_t*)positions);  //TODO make param file
    
    // Creates a ROS Node
    rclcpp::NodeOptions options;
    node = rclcpp::Node::make_shared("cyberrunner_dynamixel", options);
    rclcpp::Subscription<cyberrunner_interfaces::msg::DynamixelVel>::SharedPtr sub = node->create_subscription<cyberrunner_interfaces::msg::DynamixelVel>("cyberrunner_dynamixel/cmd", 1, &set_dynamixel_speed);
    rclcpp::Service<cyberrunner_interfaces::srv::DynamixelReset>::SharedPtr service = node->create_service<cyberrunner_interfaces::srv::DynamixelReset>("cyberrunner_dynamixel/reset", &reset_dynamixel);
    rclcpp::Service<cyberrunner_interfaces::srv::DynamixelTest>::SharedPtr test_service = node->create_service<cyberrunner_interfaces::srv::DynamixelTest>("cyberrunner_dynamixel/test", &test_dynamixel);
    se_callback_group = node->create_callback_group(rclcpp::CallbackGroupType::MutuallyExclusive);

    // Start spinning the node in a multi-threaded executor so that we can use multiple callback_groups
    rclcpp::executors::MultiThreadedExecutor executor;
    executor.add_node(node);
    executor.spin();

    rclcpp::shutdown();
    return 0;
}
