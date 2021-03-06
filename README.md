Initial robot setup:

GO TO ROBOT -> 
	- Program Robot
	- Load Program
	- External Control


To set up grasping:

Note: run these command in different tabs (open new tabs with `shift` + `ctrl` + `t`)
Note 2: Use workspace 6 (jason) for proper catkin sourcing (ws is new_ws)

METHOD 1 (launch file):

	1. Run `roslaunch push_grasp grasp_demo.launch` to launch all setup nodes together

METHOD 2 (Manually boot each node):

	1. (Optional) Run `roscore`, to open the master in its own tab

	2. Unplug and replug the camera (May not boot otherwise)

	3. Run `roslaunch camera_driver realsense_driver.launch` to boot camera

	4. Run `roslaunch ur_robot_driver ur5_bringup.launch robot_ip:=10.0.0.2` to boot robot drivers

	6. Press 'Play' on the UR5 pendant

	7. Run `roslaunch ur5_moveit_config ur5_moveit_planning_execution.launch` to launch planning (needed for robot visualization in rviz)

	8. Run `rviz -d ~/new_ws/src/push_grasp/cfg/grasp_scene.rviz`

	<!-- 9. Run `rosrun push_grasp pcl_preprocess_node.py' -->

	<!-- 9. Run `roslaunch grasp_executor pcl2_assembler.launch` to initialize point cloud stitcher helper -->

	9. Run `roslaunch agile_grasp2 robot_detect_grasps.launch` to launch agile grasp

	10. Run `rosrun robotiq_ft_sensor rq_sensor'

	5. Run `rosrun robotiq_2f_gripper_control Robotiq2FGripperRtuNode.py /dev/ttyUSB0` to initialize gripper communication

	rosrun obj_recognition obj_recognition_segmentation

	rqt_plot

	<!-- 11. Run `rosrun push_grasp pcl_stitcher_service.py` to initialize point cloud detector -->

	plug in arudino run `dmesg`
	Look at most recent output should be sometihng like:
	```
	[ 6856.027452] usb 3-1: New USB device strings: Mfr=1, Product=2, SerialNumber=220
	[ 6856.027454] usb 3-1: Manufacturer: Arduino (www.arduino.cc)
	[ 6856.027455] usb 3-1: SerialNumber: 75935313836351201031
	[ 6856.029177] cdc_acm 3-1:1.0: ttyACM0: USB ACM device
	```
	run `sudo chmod a+rw /dev/ttyACM*` to get permissions.  

	Change /dev/ttyACM** in push_grasp.py 



When ready to run grasp code:

1. Run `rosrun push_grasp push_grasp.py` to run grasping node



Notable changes made:

To agile_grasp2:

	1. Added: 
		`// Update the workspace from rosparam server
		std::vector<double> workspace_temp;
		ros::param::get("/detect_grasps/workspace", workspace_temp);
		for (double i: workspace_temp){
		std::cout << i << ' ';
		}
		std::cout << std::endl << "Updated workspace!" << std::endl;`
	to code before workspace is used (in grasp_detector.cpp), along with relevant imports:
		`#include <vector>
		#include <ros/ros.h>'
	This allows the workspace to be updated by changing the workspace ROS parameter
	