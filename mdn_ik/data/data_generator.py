# pinocchio 
import pinocchio

# data generating config
# from data import DataGenConfig
try:
    from data_config import DataGenConfig
except:
    from data.data_config import DataGenConfig


# python
import numpy as np
from scipy.spatial.transform import Rotation as R
from os.path import dirname, abspath, join

def gen_rand_config(lower_limit: np.ndarray, upper_limit: np.ndarray) -> np.ndarray:
    return np.random.uniform(low=lower_limit, high=upper_limit)

def gen_rand_exp_config(lower_limit: np.ndarray, upper_limit: np.ndarray) -> np.ndarray:
    jmax = [ 0.46026527, 0.66581495, -0.3606656, -1.87172568, 0.06517799, 2.99494907, 2.60757]
    jmin = [ 0.30369494, 0.48949512, -0.51267991, -2.27541177,  0.06016369,  2.41499245,  1.85996451]
    return np.random.uniform(low=jmin, high=jmax)


def generate_data():
    # model paths
    pinocchio_model_dir = dirname(dirname(str(abspath(__file__)))) 
    model_path = pinocchio_model_dir + "/resources/" + DataGenConfig.ROBOT
    urdf_path = model_path + "/urdf/"+DataGenConfig.ROBOT_URDF
    # setup robot model and data
    model = pinocchio.buildModelFromUrdf(urdf_path)
    data = model.createData()
    # setup end effector
    ee_name = DataGenConfig.EE_NAME
    ee_link_id = model.getFrameId(ee_name)
    # joint limits (from urdf)
    lower_limit = np.array(model.lowerPositionLimit)
    upper_limit = np.array(model.upperPositionLimit)
    
    # setting up file writing
    file_name = DataGenConfig.OUT_FILE_NAME 
    file = open(file_name, "w")
    file.write("Pose\tConfiguration\n")

    num_data = DataGenConfig.NUM_DATA
    # data generating loop
    for i in range(0):#num_data):
        # generating feature and label
        config = gen_rand_config(lower_limit, upper_limit)
        pinocchio.framesForwardKinematics(model, data, config)
        pose = pinocchio.SE3ToXYZQUAT(data.oMf[ee_link_id])
        # converting quaternion to euler angle 
        if DataGenConfig.IS_QUAT == False:
            rotation = R.from_quat(list(pose[3:]))
            rotation_euler = rotation.as_euler("xyz")
            pose = np.concatenate((pose[0:3],rotation_euler))
        # annoying string manipulation for saving in text file
        # if we only care about a subset of the total chain
        config = config[:DataGenConfig.JOINT_DIMS]
        str_pose = [str(i) for i in pose]
        str_config = [str(i) for i in config]
        file.write(",".join(str_pose) + "," + ",".join(str_config) + "\n")
    """
    for i in range(num_data*1):
        # generating feature and label
        config = gen_rand_exp_config(lower_limit, upper_limit)
        pinocchio.framesForwardKinematics(model, data, config)
        pose = pinocchio.SE3ToXYZQUAT(data.oMf[ee_link_id])
        # converting quaternion to euler angle 
        if DataGenConfig.IS_QUAT == False:
            rotation = R.from_quat(list(pose[3:]))
            rotation_euler = rotation.as_euler("xyz")
            pose = np.concatenate((pose[0:3],rotation_euler))
        # annoying string manipulation for saving in text file
        # if we only care about a subset of the total chain
        config = config[:DataGenConfig.JOINT_DIMS]
        str_pose = [str(i) for i in pose]
        str_config = [str(i) for i in config]
        file.write(",".join(str_pose) + "," + ",".join(str_config) + '\n')
    """

    # load expert trajectory
    j_trajs = np.load('data/franka_panda_insertion_logs/experts/expert_joints_poses.npy', allow_pickle=True)
    joints = np.vstack(j_trajs)
    jmax = np.amax(joints, axis=0)
    jmin = np.amin(joints, axis=0)
    jmean = (jmax+jmin)/2.

    for _ in range(int(num_data/joints.shape[0])):
        for config in joints:
            # generating feature and label
            pinocchio.framesForwardKinematics(model, data, config)
            pose = pinocchio.SE3ToXYZQUAT(data.oMf[ee_link_id])
            # converting quaternion to euler angle 
            if DataGenConfig.IS_QUAT == False:
                rotation = R.from_quat(list(pose[3:]))
                rotation_euler = rotation.as_euler("xyz")
                pose = np.concatenate((pose[0:3],rotation_euler))
            # annoying string manipulation for saving in text file
            # if we only care about a subset of the total chain
            config = config[:DataGenConfig.JOINT_DIMS]
            str_pose = [str(i) for i in pose]
            str_config = [str(i) for i in config]
            file.write(",".join(str_pose) + "," + ",".join(str_config) + "\n")

    for _ in range(int(num_data/joints.shape[0])):
        for config in joints:
            jdelt = np.random.uniform(-1.5,1.5, size=jmean.shape)*(jmax - jmean)
            config = config + jdelt
            # generating feature and label
            pinocchio.framesForwardKinematics(model, data, config)
            pose = pinocchio.SE3ToXYZQUAT(data.oMf[ee_link_id])
            # converting quaternion to euler angle 
            if DataGenConfig.IS_QUAT == False:
                rotation = R.from_quat(list(pose[3:]))
                rotation_euler = rotation.as_euler("xyz")
                pose = np.concatenate((pose[0:3],rotation_euler))
            # annoying string manipulation for saving in text file
            # if we only care about a subset of the total chain
            config = config[:DataGenConfig.JOINT_DIMS]
            str_pose = [str(i) for i in pose]
            str_config = [str(i) for i in config]
            file.write(",".join(str_pose) + "," + ",".join(str_config) + "\n")

    for _ in range(int(num_data/joints.shape[0])):
        for config in joints:
            jdelt = np.random.uniform(-1.5,1.5, size=jmean.shape)*(jmax - jmean)
            config = jmean + jdelt
            # generating feature and label
            pinocchio.framesForwardKinematics(model, data, config)
            pose = pinocchio.SE3ToXYZQUAT(data.oMf[ee_link_id])
            # converting quaternion to euler angle 
            if DataGenConfig.IS_QUAT == False:
                rotation = R.from_quat(list(pose[3:]))
                rotation_euler = rotation.as_euler("xyz")
                pose = np.concatenate((pose[0:3],rotation_euler))
            # annoying string manipulation for saving in text file
            # if we only care about a subset of the total chain
            config = config[:DataGenConfig.JOINT_DIMS]
            str_pose = [str(i) for i in pose]
            str_config = [str(i) for i in config]
            file.write(",".join(str_pose) + "," + ",".join(str_config) + "\n")


    # close file buffer
    file.close()


#if __name__ == "__main__":
#    generate_data()
