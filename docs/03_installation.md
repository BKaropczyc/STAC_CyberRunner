Installation
=====

These are steps to install the CyberRunner software stack on **Ubuntu 24.04**

---

If using a **Virtual Machine**, create the VM using the Ubuntu 24.04 .iso and customize the settings:
- 6 Processors
- 8,192 MB RAM
- 40GB Hard drive
- Disable Side Mitigations
- UEFI Firmware emulation
	
## Install [Ubuntu 24.04](https://ubuntu.com/download/desktop)

- Performed update during installation
- Default selection of apps
- Installed:
    - 3rd party software for HW
    - Support for additional media formats
- Don't require a password to login
- Restart after installation
- Run through setup
	- Don't share system data
	- Skip other settings
- Run Software Updater and install all updates
- Restart to finish updates
- Re-ran Software Updater to ensure there were no more updates
	
## Check GPU driver

If applicable, verify the driver for the GPU was installed and is working correctly:

	nvidia-smi
	
> ***NOTE***: We had to use the open-source driver to get our Nvidia 5070 GPU to show up

## Install ROS 2, [Jazzy Jalisco](https://docs.ros.org/en/jazzy/Installation/Ubuntu-Install-Debs.html):

```
locale  # check for UTF-8
sudo apt update && sudo apt install locales
sudo locale-gen en_US en_US.UTF-8
sudo update-locale LC_ALL=en_US.UTF-8 LANG=en_US.UTF-8
export LANG=en_US.UTF-8
locale  # verify settings
```

```
sudo apt install software-properties-common
sudo add-apt-repository universe
```

```
sudo apt update && sudo apt install curl -y
export ROS_APT_SOURCE_VERSION=$(curl -s https://api.github.com/repos/ros-infrastructure/ros-apt-source/releases/latest | grep -F "tag_name" | awk -F\" '{print $4}')
curl -L -o /tmp/ros2-apt-source.deb "https://github.com/ros-infrastructure/ros-apt-source/releases/download/${ROS_APT_SOURCE_VERSION}/ros2-apt-source_${ROS_APT_SOURCE_VERSION}.$(. /etc/os-release && echo $VERSION_CODENAME)_all.deb"
sudo apt install /tmp/ros2-apt-source.deb
```

```
sudo apt update && sudo apt install ros-dev-tools
```

```
sudo apt update
sudo apt upgrade
sudo apt install ros-jazzy-desktop
```

```
echo "source /opt/ros/jazzy/setup.bash" >> ~/.bashrc
echo "export ROS_DOMAIN_ID=25" >> ~/.bashrc
echo "export ROS_AUTOMATIC_DISCOVERY_RANGE=LOCALHOST" >> ~/.bashrc
```

Start a new terminal window...

```
printenv | grep -i ROS
```

Verify that ROS environment variables are declared

## Create the ROS 2 workspace and Python virtual environment
```
cd ~
mkdir cyberrunner_ws && cd cyberrunner_ws && mkdir src && cd src
git clone https://github.com/BKaropczyc/STAC_CyberRunner.git
cd ..
```

```
sudo apt install python3-full
python3 -m venv --system-site-packages virtual_env
source virtual_env/bin/activate
```

## Install Python dependencies

```
pip install --upgrade pip
```

```
pip install --upgrade "jax[cuda12]"       # If you have a CUDA GPU installed
pip install --upgrade jax                 # If you only have a CPU (e.g., on a virtual machine)
```

> **Optional**: You can check whether JAX can access your GPU with:
> ```
> python
> import jax
> jax.devices()   # Should return a CUDA device if GPU is available
> exit()
> ```

```
pip install -e src/STAC_CyberRunner/dreamerv3
```

## Install ROS dependencies
```
deactivate        # To exit the virtual environment
sudo rosdep init
rosdep update
rosdep install --from-paths src -y --ignore-src
```

## Build and install the ROS packages
```
source virtual_env/bin/activate     # To activate the virtual environment
python -m colcon build --symlink-install
source install/setup.bash
```

## Setup the Dynamixel node to not require a root password

Create a new shell script for setting the USB port latency:
```
sudo editor /usr/local/sbin/set_usb_latency.sh
```

Copy & paste the contents of `cyberrunner_dynamixel/utils/set_usb_latency.sh` into this document and save it.

Make the file executable:
```
sudo chmod a+x /usr/local/sbin/set_usb_latency.sh
```

Add this file to the sudoers list:

```
sudo visudo -f /etc/sudoers.d/usblatency
```
Put the following contents in this file and save it:

```
%dialout ALL=PASSWD: ALL, NOPASSWD: /usr/local/sbin/set_usb_latency.sh
```

Make sure the current user is part of the dialout group:
```
sudo usermod -aG dialout <your_user_name>
```

Reboot the machine and test this setup with:

```
sudo /usr/local/sbin/set_usb_latency.sh
```

It should indicate that is is setting the latency for `USB0`, and should **NOT** retuire a root password to execute

## Install other required packages

```
sudo apt install ffmpeg
```

---

## Common Commands:

### To activate the Python virtual environment:

```
cd ~/cyberrunner_ws
source virtual_env/bin/activate
```

### To "source" the ROS workspace:

```
cd ~/cyberrunner_ws
python -m colcon build --symlink-install    (only if needed)
source install/setup.bash
```

### To begin training from a full reboot:
```
cd ~/cyberrunner_ws
source install/setup.bash
ros2 launch cyberrunner_dreamer train_launch.xml
```

---
