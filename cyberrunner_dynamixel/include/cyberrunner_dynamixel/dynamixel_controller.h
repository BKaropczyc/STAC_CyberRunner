#ifndef DYNAMIXEL_CONTROLLER_H
#define DYNAMIXEL_CONTROLLER_H

int dynamixel_init(const char* port, int num_dynamixel, uint8_t* dynamixel_ids, int baudrate);
int dynamixel_set_operating_mode(int num_dynamixel, uint8_t* dynamixel_ids, uint8_t operating_mode);
int dynamixel_set_torque_enable(int num_dynamixel, uint8_t* dynamixel_ids, uint8_t torque_enable);
int dynamixel_set_goal_current(int num_dynamixel, uint8_t* dynamixel_ids, int16_t* currents);
int dynamixel_set_goal_velocity(int num_dynamixel, uint8_t* dynamixel_ids, int32_t* velocities);

#endif
