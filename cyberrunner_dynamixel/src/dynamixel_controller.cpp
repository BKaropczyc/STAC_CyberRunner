#include <fcntl.h>
#include <termios.h>
#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>
#include <unistd.h>
#include <sys/time.h>
#include "dynamixel_sdk/dynamixel_sdk.h"

#include "cyberrunner_dynamixel/dynamixel_controller.h"
#include "cyberrunner_dynamixel/error_handling.h"

// Control table address, these might be different for different models.
#define ADDR_OPERATING_MODE             11
#define ADDR_XL_TORQUE_ENABLE           64
#define ADDR_XL_PRESENT_VELOCITY        128
#define ADDR_XL_TEMPERATURE_LIMIT       31
#define ADDR_XL_PRESENT_POSITION        132
#define ADDR_XL_VELOCITY_KI             76
#define ADDR_XL_VELOCITY_KP             78
#define ADDR_XL_GOAL_CURRENT            102
#define ADDR_XL_GOAL_VELOCITY           104
#define ADDR_XL_HARDWARE_ERROR_STATUS   70  // Hardware error packet retrieval
#define ADDR_XL_PWM_OUTPUT              100 // Address for PWM output

// Protocol version
#define PROTOCOL_VERSION                2.0 // See which protocol version is used in the Dynamixel

// Values
#define XL_TEMPERATURE_LIMIT            50  // Maximum Temperature Allowable for Dynamixel
#define CURRENT_CONTROL_MODE            0
#define VELOCITY_CONTROL_MODE           1
#define TORQUE_ENABLE                   1   // Value for enabling the torque
#define TORQUE_DISABLE                  0   // Value for disabling the torque
#define I_GAIN_VALUE                    400
#define P_GAIN_VALUE                    40

// Private function declarations
void print_hardware_error_status(uint8_t dynamixel_id);

dynamixel::PortHandler *port_handler = NULL;
dynamixel::PacketHandler *packet_handler = NULL;

using namespace dynamixel;

int dynamixel_init(const char* port, int num_dynamixel, uint8_t* dynamixel_ids, int baudrate)
{
    if(port_handler == NULL || packet_handler == NULL)
    {
        // Minimize latency on USB communication
        char device[10];
        char cmd[256];
        sscanf(port, "/dev/%s", device);
        sprintf(cmd, "sudo /usr/local/sbin/set_usb_latency.sh %s", device);
        printf("Executing command: %s\n", cmd);
        system(cmd);

        // Initialize PortHandler Structs
        port_handler = PortHandler::getPortHandler(port);

        // Initialize PacketHandler Structs
        packet_handler = PacketHandler::getPacketHandler(PROTOCOL_VERSION);

        // Open port
        if (port_handler->openPort())
        {
            printf("Succeeded to open the port!\n");
        }
        else
        {
            printf("Failed to open the port!\n");
            return -1;
        }

        // Set port baudrate
        if (port_handler->setBaudRate(baudrate))
        {
            printf("Succeeded to change the baudrate!\n");
        }
        else
        {
            printf("Failed to change the baudrate!\n");
            return -1;
        }
    }

    // Set the servos to current control mode
    if (dynamixel_set_operating_mode(num_dynamixel, dynamixel_ids, CURRENT_CONTROL_MODE) == 0)
        printf("Succeeded to set servos to current control mode!\n");
    else
        return -1;

    for(int i = 0; i < num_dynamixel; i++)
        printf("Dynamixel (ID: %d) has been successfully connected \n", dynamixel_ids[i]);

    return 0;
}

int dynamixel_set_operating_mode(int num_dynamixel, uint8_t* dynamixel_ids, uint8_t operating_mode)
{
    int dxl_comm_result = COMM_TX_FAIL;
    uint8_t dxl_error = 0;

    // Disable torque
    if (dynamixel_set_torque_enable(num_dynamixel, dynamixel_ids, TORQUE_DISABLE) != 0)
        return -1;

    // Set the operating mode
    for(int i = 0; i < num_dynamixel; i++)
    {
        dxl_comm_result = packet_handler->write1ByteTxRx(port_handler, dynamixel_ids[i], ADDR_OPERATING_MODE, operating_mode, &dxl_error);
        if (dxl_comm_result != COMM_SUCCESS)
        {
            printf("Failed to set Operating Mode for Dynamixel ID %d: %s\n", dynamixel_ids[i], packet_handler->getTxRxResult(dxl_comm_result));
            return -1;
        }
        else if (dxl_error != 0)
        {
            printf("Error setting Operating Mode for Dynamixel ID %d: %s\n", dynamixel_ids[i], packet_handler->getRxPacketError(dxl_error));
            return -1;
        }
    }

    // Reset the corresponding goal values
    if (operating_mode == CURRENT_CONTROL_MODE)
    {
        // Set all currents to 0
        int16_t* currents = (int16_t*)calloc(num_dynamixel, sizeof(int16_t));
        dynamixel_set_goal_current(num_dynamixel, dynamixel_ids, currents);
        free(currents);
    }
    else if (operating_mode == VELOCITY_CONTROL_MODE)
    {
        // Set all velocities to 0
        int32_t* velocities = (int32_t*)calloc(num_dynamixel, sizeof(int32_t));
        dynamixel_set_goal_velocity(num_dynamixel, dynamixel_ids, velocities);
        free(velocities);
    }
    else
    {
        printf("Error: Unknown operating mode! Can't reset goal values.\n");
        return -1;
    }

    // Enable torque
    if (dynamixel_set_torque_enable(num_dynamixel, dynamixel_ids, TORQUE_ENABLE) != 0)
        return -1;

    return 0;
}

int dynamixel_set_torque_enable(int num_dynamixel, uint8_t* dynamixel_ids, uint8_t torque_enable)
{
    int dxl_comm_result = COMM_TX_FAIL;
    uint8_t dxl_error = 0;

    for(int i = 0; i < num_dynamixel; i++)
    {
        // Update Torque Enable value
        dxl_comm_result = packet_handler->write1ByteTxRx(port_handler, dynamixel_ids[i], ADDR_XL_TORQUE_ENABLE, torque_enable, &dxl_error);
        if (dxl_comm_result != COMM_SUCCESS)
        {
            const char* verb = (torque_enable == TORQUE_ENABLE) ? "enable" : "disable";
            printf("Failed to %s torque for Dynamixel ID %d: %s\n", verb, dynamixel_ids[i], packet_handler->getTxRxResult(dxl_comm_result));
            return -1;
        }
        if (dxl_error != 0)
        {
            const char* verb = (torque_enable == TORQUE_ENABLE) ? "enabling" : "disabling";
            printf("Error %s torque for Dynamixel ID %d: %s\n", verb, dynamixel_ids[i], packet_handler->getRxPacketError(dxl_error));
            return -1;
        }
    }

    return 0;
}

int dynamixel_set_goal_current(int num_dynamixel, uint8_t* dynamixel_ids, int16_t* currents)
{
    int dxl_comm_result = COMM_TX_FAIL;
    uint8_t dxl_error = 0;

    for(int i = 0; i < num_dynamixel; i++)
    {
        // Write goal current (2 bytes)
        dxl_comm_result = packet_handler->write2ByteTxRx(port_handler, dynamixel_ids[i], ADDR_XL_GOAL_CURRENT, currents[i], &dxl_error);
        if (dxl_comm_result != COMM_SUCCESS)
        {
            printf("Failed to write goal current for Dynamixel ID %d: %s\n", dynamixel_ids[i], packet_handler->getTxRxResult(dxl_comm_result));
            continue;
        }
        if (dxl_error != 0)
        {
            printf("Error writing goal current for Dynamixel ID %d: %s\n", dynamixel_ids[i], packet_handler->getRxPacketError(dxl_error));
            print_hardware_error_status(dynamixel_ids[i]);

            // Check for overload error and handle it
            uint8_t hardware_error_status = 0;
            packet_handler->read1ByteTxRx(port_handler, dynamixel_ids[i], ADDR_XL_HARDWARE_ERROR_STATUS, &hardware_error_status, &dxl_error);
            if (hardware_error_status & 0x10) // Bit 4 indicates overload error
            {
                printf("Overload error detected on Dynamixel ID %d. Disabling torque.\n", dynamixel_ids[i]);
                packet_handler->write1ByteTxRx(port_handler, dynamixel_ids[i], ADDR_XL_TORQUE_ENABLE, TORQUE_DISABLE, &dxl_error);
                continue;
            }
        }
    }

    return 0;
}

int dynamixel_set_goal_velocity(int num_dynamixel, uint8_t* dynamixel_ids, int32_t* velocities)
{
    int dxl_comm_result = COMM_TX_FAIL;
    uint8_t dxl_error = 0;

    for(int i = 0; i < num_dynamixel; i++)
    {
        // Write goal velocity (4 bytes)
        dxl_comm_result = packet_handler->write4ByteTxRx(port_handler, dynamixel_ids[i], ADDR_XL_GOAL_VELOCITY, velocities[i], &dxl_error);
        if (dxl_comm_result != COMM_SUCCESS)
        {
            printf("Failed to write goal velocity for Dynamixel ID %d: %s\n", dynamixel_ids[i], packet_handler->getTxRxResult(dxl_comm_result));
            continue;
        }
        if (dxl_error != 0)
        {
            printf("Error writing goal velocity for Dynamixel ID %d: %s\n", dynamixel_ids[i], packet_handler->getRxPacketError(dxl_error));
            print_hardware_error_status(dynamixel_ids[i]);

            // Check for overload error and handle it
            uint8_t hardware_error_status = 0;
            packet_handler->read1ByteTxRx(port_handler, dynamixel_ids[i], ADDR_XL_HARDWARE_ERROR_STATUS, &hardware_error_status, &dxl_error);
            if (hardware_error_status & 0x10) // Bit 4 indicates overload error
            {
                printf("Overload error detected on Dynamixel ID %d. Disabling torque.\n", dynamixel_ids[i]);
                packet_handler->write1ByteTxRx(port_handler, dynamixel_ids[i], ADDR_XL_TORQUE_ENABLE, TORQUE_DISABLE, &dxl_error);
                continue;
            }
        }
    }

    return 0;
}

void print_hardware_error_status(uint8_t dynamixel_id)
{
    uint8_t hardware_error_status = 0;
    uint8_t dxl_error = 0;
    int dxl_comm_result = packet_handler->read1ByteTxRx(port_handler, dynamixel_id, ADDR_XL_HARDWARE_ERROR_STATUS, &hardware_error_status, &dxl_error);

    if (dxl_comm_result != COMM_SUCCESS)
    {
        printf("Failed to read hardware error status for Dynamixel ID %d: %s\n", dynamixel_id, packet_handler->getTxRxResult(dxl_comm_result));
    }
    else if (dxl_error != 0)
    {
        printf("Hardware error status for Dynamixel ID %d: %s\n", dynamixel_id, packet_handler->getRxPacketError(dxl_error));
    }
    else
    {
        printf("Hardware error status for Dynamixel ID %d: 0x%02X\n", dynamixel_id, hardware_error_status);
    }
}
