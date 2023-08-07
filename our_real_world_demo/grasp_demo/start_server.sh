aruco_dir="/home/wenke/hand_eye"
graspnet_dir="/home/wenke/workspace/graspnet-baseline"
robot_dir="/home/wenke/workspace/our_real_world_demo/grasp_demo"
conda_path="~/miniconda3/etc/profile.d/conda.sh"

echo "start the camera server"

terminator --new-tab -e "roscore" & sleep 1

terminator --new-tab -e "cd $aruco_dir; python src/release_true_realsense_intr.py"

terminator --new-tab -e "cd $aruco_dir; source devel/setup.bash; roslaunch handeye-calib aruco_without_realsense.launch"

echo "start the graspnet server"

terminator --new-tab -e "source $conda_path; conda activate graspnet; cd $graspnet_dir; python demo.py --checkpoint_path checkpoint-rs.tar"

echo "start the robotic server"

terminator --new-tab -e "source ~/.bashrc; source $conda_path; conda activate grasp; cd $robot_dir; python camera_server.py"

terminator --new-tab -e "source ~/.bashrc; source $conda_path; conda activate grasp;  cd $robot_dir; python server.py"
