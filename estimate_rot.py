import numpy as np
from scipy import io, linalg
from quaternion import Quaternion
import math
import matplotlib.pyplot as plt


class ukfPipeline:
    """
    The variables will follow Edgar Craft's convention and where its not mentioned
    """

    DEG2RAD = math.pi / 180

    def __init__(self, accel, gyro, T):
        # Strat for bias -> set sensitivity to 1 and then take mean of first 700 readings
        # Then do sensitivity by making sure the peaks of vicon and acc matches
        self.acc_sens = np.array([31, 31, 29]).reshape(-1, 1)  # mV/g
        self.acc_bias = np.array([510.8047, 500.9942, 500.81]).reshape(-1, 1)
        self.gyro_sens = np.array([3.5, 3.15, 3.05]).reshape(-1, 1)
        self.gyro_bias = np.array([369.68571429, 373.57142857, 370.37285714]).reshape(
            -1, 1
        )
        self.accel, self.gyro = self.parseIMU(
            accel.astype(np.float64), gyro.astype(np.float64)
        )
        self.timesteps = T
        print(f"[UKF] Sensor data loaded and callibrated")

        # Define all the matrices

        # This is process noise
        self.R = np.diag([1, 1, 1, 1, 1, 1])*0
        # This is filter covariance -> sigma_k|k
        self.filterCov = np.diag([1, 1, 1, 1, 1, 1])*0
        # Initial States are 0 degree angles -> mu_k|k
        self.state = np.array([1, 0, 0, 0, 0, 0, 0]).reshape(-1, 1)

    def parseIMU(self, accel, gyro):
        return self.parseAccelerometer(accel), self.parseGyro(gyro)

    def parseAccelerometer(self, accel):
        rawReading = (accel - self.acc_bias) * (3300 / (1023 * self.acc_sens)) * 9.81
        # Inverting first 2 rows according to the datasheet
        rawReading[:2, :] *= -1
        return rawReading

    def convertAcc2Angle(self):
        # Pitch is tan2(y/z)
        roll = np.arctan2(self.accel[1, :].flatten(), self.accel[2, :].flatten())
        pitch = np.arctan2(
            -self.accel[0, :].flatten(),
            np.linalg.norm(self.accel[1:, :], axis=0).flatten(),
        )
        return roll, pitch

    def parseGyro(self, gyro):
        gyroSens = self.gyro_sens[[2, 0, 1]]
        gyroBias = self.gyro_bias[[2, 0, 1]]
        rawReading = (
            (gyro - gyroBias) * (3300 / (1023 * gyroSens)) * ukfPipeline.DEG2RAD
        )
        return rawReading[[1, 2, 0], :]  # you learn something new everyday

    def calibrationOutput(self):
        roll, pitch = self.convertAcc2Angle()
        roll, pitch, yaw = self.gyro[0, :], self.gyro[1, :], self.gyro[2, :]
        return roll, pitch, yaw

    def quaternionMeanGD(self, quats):
        #meaning of quats is that it is quaternion sigma points 4x2n
        e = np.zeros_like(quats)
        vectorE = np.zeros((3,quats.shape[1]))
        itr = 0 #max Iter

        qt_bar = Quaternion(float(self.state[0]),self.state[1:4].flatten())

        qi = Quaternion()
        while itr<50:
            for i in range(quats.shape[1]):
                qi.q = quats[:,i].flatten()
                e[:,i] = qi.__mul__(qt_bar.inv()).q
                vectorE[:,i] = qi.__mul__(qt_bar.inv()).axis_angle().flatten()
            eBar = vectorE.mean(axis=1)
            qt_bar.from_axis_angle(eBar)
            itr += 1
            if np.linalg.norm(eBar)<0.01:
                break
        return qt_bar.q,vectorE
    
    def stateParams(self, Yi):
        quatMean,errVec = self.quaternionMeanGD(Yi[:4,:])
        omegaMean = Yi[4:].mean(axis=1).reshape(-1,1)
        mu_k = np.vstack((quatMean.reshape(-1,1),omegaMean))
        cov = self.stateCov(errVec.mean(axis=1),errVec,Yi)        
        return mu_k,cov

    def stateCov(self,ebar,errVec,Yi):
        ebar = ebar.reshape(-1,1)
        omegaMean = Yi[4:].mean(axis=1).reshape(-1,1)
        mat = (errVec-ebar) #3x6
        omegaCov = Yi[4:] - omegaMean
        
        cov = np.vstack((mat,omegaCov))
        return (cov@cov.T)/Yi.shape[1]

    def propogateStep(self, dt):
        # Propogating Dynamics

        # Modifying Filter Covariance by adding Process Noise and timestep
        self.filterCov = self.filterCov + self.R * dt

        # Calculating sigma points with Quaternion inclusion
        sgmaPts,_ = self.calculateSigmaPoints(self.filterCov, self.state)

        # Doing Unscented Transform by passing through non linear function
        # Temporary Quaternions for propogating dynamics
        quatSigma = Quaternion()
        quatDump = Quaternion()
        Yi = np.zeros_like(sgmaPts)
        for i in range(sgmaPts.shape[1]):
            quatSigma.q = sgmaPts[:4, i]
            quatDump.q = self.state[:4].reshape(-1,)
            Yi[:4, i] = quatDump.__mul__(quatSigma).q
        Yi[4:, :] += sgmaPts[4:, :]

        #Finding the mean and covariances
        self.state,self.filterCov = self.stateParams(Yi)

    def calculateSigmaPoints(self, Cov, mean):
        spts = np.zeros((Cov.shape[0], Cov.shape[0] * 2))
        spts[:, :6] = linalg.sqrtm(Cov) * np.sqrt(Cov.shape[0])
        spts[:, 6:] = linalg.sqrtm(Cov) * np.sqrt(Cov.shape[0])
        sptsQuat = self.cnvrtSpts2Quat(spts)
        sptsQuat[:, :6] = mean + sptsQuat[:, :6]
        sptsQuat[:, :6] = mean - sptsQuat[:, :6]
        return sptsQuat,spts

    def cnvrtSpts2Quat(self, spts):
        sptsQuat = np.zeros((spts.shape[0] + 1, spts.shape[1]))
        tmpQuat = Quaternion()
        for i in range(spts.shape[1]):
            tmpQuat.from_axis_angle(spts[:3, i])
            sptsQuat[:4, i] = tmpQuat.q
        sptsQuat[4:, :] = spts[3:, :]
        return sptsQuat

    def measurementStep(self):
        pass

    def runPipeline(self):
        stateVector = self.state.copy()
        nSteps = self.accel.shape[1] - 1
        for i in range(nSteps):
            dt = self.timesteps[i + 1] - self.timesteps[i]
            self.propogateStep(dt)
            self.measurementStep(dt)
            stateVector = np.column_stack((stateVector, self.state))

        return stateVector


def viconAngle(ang, T):
    return (ang[1:] - ang[:-1]) / 0.01


def estimate_rot(data_num=1):
    # load data
    imu = io.loadmat("imu/imuRaw" + str(data_num) + ".mat")
    vicon = io.loadmat("vicon/viconRot" + str(data_num) + ".mat")
    accel = imu["vals"][0:3, :]
    gyro = imu["vals"][3:6, :]
    T = (imu["ts"]).flatten()

    sol = ukfPipeline(accel, gyro, T)
    stateVector = sol.runPipeline()
    roll, pitch, yaw = stateVector


    # roll, pitch, yaw are numpy arrays of length T
    return roll, pitch, yaw


if __name__ == "__main__":
    estimate_rot()






    # vicon2Sens = np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]])
    # r = []
    # quat = Quaternion()
    # for i in range(vicon["rots"].shape[-1]):
    #     R = vicon["rots"][:, :, i].reshape(3, 3)
    #     quat.from_rotm(R)
    #     ang = quat.euler_angles()
    #     r.append(float(ang[2]))
    # r = np.array(r)