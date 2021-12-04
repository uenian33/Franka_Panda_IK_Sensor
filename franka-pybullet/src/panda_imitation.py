import pybullet as p
import sys
from pathlib import Path
sys.path.append(Path(__file__).parent.absolute())
import transformation
import numpy as np



class Panda:
    def __init__(self, stepsize=1e-3, realtime=0):
        self.t = 0.0
        self.stepsize = stepsize
        self.realtime = realtime

        self.control_mode = "torque" 

        self.position_control_gain_p = [0.01,0.01,0.01,0.01,0.01,0.01,0.01]
        self.position_control_gain_d = [1.0,1.0,1.0,1.0,1.0,1.0,1.0]
        self.max_torque = [100,100,100,100,100,100,100]
        self.init_torque = [ 0.38580915,  0.49720666, -0.44974689, -2.2750547 ,  0.06505849,2.99288612,  2.3980645 ]


        self.action_max = []


        # connect pybullet
        p.connect(p.GUI)
        p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
        p.resetDebugVisualizerCamera(cameraDistance=1.5, cameraYaw=30, cameraPitch=-20, cameraTargetPosition=[0, 0, 0.5])

        p.resetSimulation()
        p.setTimeStep(self.stepsize)
        p.setRealTimeSimulation(self.realtime)
        p.setGravity(0,0,-9.81)

        # load models
        p.setAdditionalSearchPath(str(Path(__file__).parent.parent.absolute())+"/models")

        self.plane = p.loadURDF("plane/plane.urdf",
                                useFixedBase=True)
        p.changeDynamics(self.plane,-1,restitution=.95)

        self.robot = p.loadURDF("panda/panda.urdf",
                                useFixedBase=True,
                                flags=p.URDF_USE_SELF_COLLISION)
        
        # robot parameters
        self.dof = p.getNumJoints(self.robot) - 1 # Virtual fixed joint between the flange and last link
        if self.dof != 7:
            raise Exception('wrong urdf file: number of joints is not 7')

        self.joints = []
        self.q_min = []
        self.q_max = []
        self.target_pos = []
        self.target_torque = []

        for j in range(self.dof):
            joint_info = p.getJointInfo(self.robot, j)
            self.joints.append(j)
            self.q_min.append(joint_info[8])
            self.q_max.append(joint_info[9])
            self.target_pos.append((self.q_min[j] + self.q_max[j])/2.0)
            self.target_torque.append(0.)

        self.reset()

    def reset(self):
        self.t = 0.0        
        self.control_mode = "torque"
        """
        for j in range(self.dof):
            self.target_pos[j] = (self.q_min[j] + self.q_max[j])/2.0
            self.target_torque[j] = 0.
            p.resetJointState(self.robot,j,targetValue=self.target_pos[j])
        """
        for j in range(self.dof):
            self.target_pos[j] = (self.q_min[j] + self.q_max[j])/2.0
            self.target_torque[j] = 0.
            p.resetJointState(self.robot,j,targetValue=self.init_torque[j])

        self.resetController()
       
        return self.getEEStatesEuler()

    def setLimitations(self,
                      demo_path='../franka_panda_insertion_logs/experts/expert_euler_poses.npy'):
        cat_trajs = np.load(demo_path, allow_pickle=True)
        self.state_max = np.amax(np.vstack(cat_trajs), axis=0)
        self.state_min = np.amin(np.vstack(cat_trajs), axis=0)
        self.state_mean = (self.state_min + self.state_max) / 2.
        self.action_max = np.amax(np.vstack(cat_trajs), axis=0) - self.state_mean

    def simStep(self):
        self.t += self.stepsize
        p.stepSimulation()

    def step(self, euler_pose, simsteps=20):
        euler_pose = np.clip(euler_pose, self.state_min,self.state_max)

        target_torque = self.solveInverseKinematics(euler_pose[:3],transformation.quaternion_from_euler(euler_pose[3], euler_pose[4], euler_pose[5]))#[1,0,0,0])
        self.setTargetPositions(target_torque)

        for i in range(simsteps):
            self.simStep()

        euler_pose = self.getEEStatesEuler()

        reward = 0

        done = False

        info = ''

        return euler_pose, reward, done, info

    def stepTorque(self, target_torque, simsteps=20):
        self.setTargetPositions(target_torque)

        for i in range(simsteps):
            self.simStep()

        euler_pose = self.getEEStatesEuler()

        return euler_pose

    def checkStates(self):
        euler_pose = self.getEEStatesEuler()
        if np.any(euler_pose>self.state_max) or np.any(euler_pose<self.state_max):
            return True
        else:
            return False

    # robot functions
    def resetController(self):
        p.setJointMotorControlArray(bodyUniqueId=self.robot,
                                    jointIndices=self.joints,
                                    controlMode=p.VELOCITY_CONTROL,
                                    forces=[0. for i in range(self.dof)])

    def setControlMode(self, mode):
        if mode == "position":
            self.control_mode = "position"
        elif mode == "torque":
            if self.control_mode != "torque":
                self.resetController()
            self.control_mode = "torque"
        else:
            raise Exception('wrong control mode')

    def setTargetPositions(self, target_pos):
        self.target_pos = target_pos
        p.setJointMotorControlArray(bodyUniqueId=self.robot,
                                    jointIndices=self.joints,
                                    controlMode=p.POSITION_CONTROL,
                                    targetPositions=self.target_pos,
                                    forces=self.max_torque,
                                    positionGains=self.position_control_gain_p,
                                    velocityGains=self.position_control_gain_d)

    def setTargetTorques(self, target_torque):
        self.target_torque = target_torque
        p.setJointMotorControlArray(bodyUniqueId=self.robot,
                                    jointIndices=self.joints,
                                    controlMode=p.TORQUE_CONTROL,
                                    forces=self.target_torque)


    def getEEStates(self):
        state_robot = p.getLinkState(self.robot, 7)
        return state_robot

    def getEEStatesEuler(self):
        state_robot = p.getLinkState(self.robot, 7)
        pos = state_robot[0]
        euler = transformation.euler_from_quaternion(state_robot[1])
        return np.array([pos[0], pos[1], pos[2], euler[0], euler[1], euler[2]])

    def getEEStatesQuaternion(self):
        state_robot = p.getLinkState(self.robot, 7)
        pos = state_robot[0]
        quat = state_robot[1]
        return np.array([pos[0], pos[1], pos[2], quat[0], quat[1], quat[2], quat[3]])

    def getJointStates(self):
        #print("!!!!!!!")
        joint_states = p.getJointStates(self.robot, self.joints)
        joint_pos = [x[0] for x in joint_states]
        joint_vel = [x[1] for x in joint_states]
        return joint_pos, joint_vel 

    def solveInverseDynamics(self, pos, vel, acc):
        return list(p.calculateInverseDynamics(self.robot, pos, vel, acc))

    def solveInverseKinematics(self, pos, ori):
        return list(p.calculateInverseKinematics(self.robot, 7, pos, ori))


if __name__ == "__main__":
    robot = Panda(realtime=1)
    while True:
        pass
