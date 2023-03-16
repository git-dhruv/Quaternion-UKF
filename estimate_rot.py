"""
@author: Dhruv Parikh
@date: 3rd March 2023
@Course: ESE 6500 Learning in Robotics
"""

import numpy as np
from scipy import io, linalg
from quaternion import Quaternion
import math


class ukfPipeline:
    """
    The variables will follow Edgar Craft's convention and where its not mentioned
    """

    DEG2RAD = math.pi / 180
    G = 9.80665

    def __init__(self, accel, gyro, T):
        # Strat for bias -> set sensitivity to 1 and then take mean of first 700 readings
        # Then do sensitivity by making sure the peaks of vicon and acc matches

        ## ---- Sensor Calibration----##
        self.acc_sens = np.array([3.3217, 3.3217, 3.3217]).reshape(-1, 1)  # mV/g
        self.acc_bias = accel[:, :50].mean(axis=1).reshape(-1, 1)
        self.acc_bias[-1] -= 98.1
        self.gyro_sens = np.array([193.55, 193.55, 193.55]).reshape(-1, 1)
        self.gyro_bias = gyro[:, :50].mean(axis=1).reshape(-1, 1)
        self.accel, self.gyro = self.parseIMU(
            accel.astype(np.float64), gyro.astype(np.float64))
        self.timesteps = T

        ## ---- Filter Setup----##
        # Define all the matrices
        # This is process noise
        self.R = np.diag([2200, 2200, 2200, 1000, 1000, 1000])
        # Observation Noise
        self.Q = np.diag([225, 225, 225, 150, 150, 225]) / 34

        # This is filter covariance -> sigma_k|k
        self.filterCov = np.diag([0, 0, 0.0, 0, 0, 0])
        # Initial States are 0 degree angles -> mu_k|k
        self.state = np.array([1, 0, 0, 0, 0, 0, 0]).reshape(-1, 1)
        print(f"[UKF] Sensor data loaded and callibrated")

    def parseIMU(self, accel, gyro):
        return self.parseAccelerometer(accel), self.parseGyro(gyro)

    def parseAccelerometer(self, accel):
        rawReading = (accel - self.acc_bias) * \
            (3300 / (1023 * self.acc_sens)) * self.G
        # Inverting first 2 rows according to the datasheet
        rawReading[:2, :] *= -1
        rawReading = self.G * rawReading / np.linalg.norm(rawReading, axis=0)
        return rawReading

    def convertAcc2Angle(self):
        # Pitch is tan2(y/z)
        roll = np.arctan2(self.accel[1, :].flatten(),
                          self.accel[2, :].flatten())
        pitch = np.arctan2(
            -self.accel[0, :].flatten(),
            np.linalg.norm(self.accel[1:, :], axis=0).flatten(),
        )
        return roll, pitch

    def parseGyro(self, gyro):
        gyroSens = self.gyro_sens
        gyroBias = self.gyro_bias
        rawReading = (gyro - gyroBias) * (3300 / (1023 * gyroSens))
        return rawReading[[1, 2, 0], :]  # you learn something new everyday

    def calibrationOutput(self):
        """
        Calibration util -> not used in filtering
        """
        roll, pitch = self.convertAcc2Angle()
        roll, pitch, yaw = self.gyro[0, :], self.gyro[1, :], self.gyro[2, :]
        return roll, pitch, yaw

    def quaternionMeanGD(self, quats):
        """
        Gradient Descent based mean calculation of quaternion
        """

        # meaning of quats is that it is quaternion sigma points 4x2n
        vectorE = np.zeros((3, quats.shape[1]))
        itr = 0  # max Iter

        # Initializing mean quaternion
        qt_bar = Quaternion(float(self.state[0]), self.state[1:4].flatten())
        while itr <= 85:
            tmp = qt_bar.inv()

            for i in range(quats.shape[1]):
                qi = Quaternion((quats[0, i]), quats[1:, i].flatten())
                vectorE[:, i] = qi.__mul__(tmp).axis_angle()
            eBar = vectorE.mean(axis=1)
            tmp = Quaternion()
            tmp.from_axis_angle(eBar)
            qt_bar = tmp.__mul__(qt_bar)
            itr += 1
            if itr == 1:
                qVec = [qt_bar]
                eBarVec = [np.linalg.norm(eBar)]
            else:
                qVec.append(qt_bar)
                eBarVec.append(np.linalg.norm(eBar))
            if np.linalg.norm(eBar) <= 0.002:
                break
            if itr == 80:
                # You cant just break on max iter and call it a day
                index = eBarVec.index(min(eBarVec))
                qt_bar = qVec[index]
                for i in range(quats.shape[1]):
                    qi = Quaternion(float(quats[0, i]), quats[1:, i].flatten())
                    vectorE[:, i] = qi.__mul__(
                        qt_bar.inv()).axis_angle().flatten()
                break

        return qt_bar.q, vectorE

    def stateParams(self, Yi):
        quatMean, errVec = self.quaternionMeanGD(Yi[:4, :])
        omegaMean = Yi[4:].mean(axis=1).reshape(-1, 1)
        mu_k = np.vstack((quatMean.reshape(-1, 1), omegaMean))
        cov = self.stateCov(errVec, Yi)
        return mu_k, cov

    def stateCov(self, errVec, Yi):
        omegaMean = Yi[4:].mean(axis=1).reshape(-1, 1)
        mat = errVec  # 3x6
        omegaCov = Yi[4:] - omegaMean
        cov = np.vstack((mat, omegaCov))
        return np.cov(cov, bias=1)

    def calculateSigmaPoints(self, Cov, mean):
        spts = np.zeros((Cov.shape[0], Cov.shape[0] * 2))
        tmp = linalg.sqrtm(Cov) * np.sqrt(Cov.shape[1])
        spts[:, :6] = tmp
        spts[:, 6:] = -tmp
        sptsQuat = self.cnvrtSpts2Quat(spts)  # 7x12
        sptsQuat[4:, :] = mean[4:] + sptsQuat[4:, :]
        muk = Quaternion(float(mean[0]), mean[1:4].flatten())
        for i in range(12):
            sptsQuat[:4, i] = muk.__mul__(
                Quaternion(float(sptsQuat[0, i]), sptsQuat[1:4, i].flatten())
            ).q
        return sptsQuat, spts

    def cnvrtSpts2Quat(self, spts):
        sptsQuat = np.zeros((spts.shape[0] + 1, spts.shape[1]))
        tmpQuat = Quaternion()
        for i in range(spts.shape[1]):
            tmpQuat.from_axis_angle(spts[:3, i])
            sptsQuat[:4, i] = tmpQuat.q
        sptsQuat[4:, :] = spts[3:, :]
        return sptsQuat

    def propogateStep(self, dt):
        # Propogating Dynamics

        # Modifying Filter Covariance by adding Process Noise and timestep
        self.filterCov = self.filterCov + self.R * dt

        # Calculating sigma points with Quaternion inclusion
        sgmaPts, _ = self.calculateSigmaPoints(self.filterCov, self.state)

        # Doing Unscented Transform by passing through non linear function
        # Temporary Quaternions for propogating dynamics
        quatDump = Quaternion(float(self.state[0]), self.state[1:4].flatten())
        Yi = np.zeros_like(sgmaPts)
        for i in range(sgmaPts.shape[1]):
            # Converting omega points to quaternion
            omegas = sgmaPts[4:, i]
            quatDump = Quaternion()
            quatDump.from_axis_angle(omegas * dt)
            # This is quat sigma
            quatSigma = Quaternion(
                float(sgmaPts[0, i]), sgmaPts[1:4, i].flatten())
            tmp = quatSigma.__mul__(quatDump)
            Yi[:4, i] = tmp.q
        Yi[4:, :] = sgmaPts[4:, :]

        # Finding the mean and covariances
        self.state, self.filterCov = self.stateParams(Yi)
        return Yi

    def measurementStep(self, Yi, observation):
        Yi, yiVec = self.calculateSigmaPoints(self.filterCov, self.state)
        # propogate sigma points using measurement models
        quats = Yi[:4, :]
        omegas = Yi[4:, :]
        Zi = np.zeros((6, Yi.shape[1]))

        g = Quaternion(0.0, [0.0, 0.0, self.G])

        for i in range(quats.shape[1]):
            q = Quaternion(float(quats[0, i]), quats[1:, i].flatten())  # sigma
            # H1 is given by
            inter = q.inv() * g * (q)
            Zi[:3, i] = inter.vec()

        # H2 is this
        Zi[3:] = omegas

        # Compute new mean and covariance
        mean = np.mean(Zi, axis=1)
        mat = Zi - mean.reshape(-1, 1)
        cov = (mat @ mat.T) / Zi.shape[1]

        # Add the observation Noise
        cov = cov + self.Q  # correct

        # Cross correlation is a different game now
        axisYi = yiVec
        cov_xy = np.zeros((6, 6))
        vectorE = np.zeros((3, 12))
        qt_bar = Quaternion(float(self.state[0]), self.state[1:4].flatten())
        for i in range(12):
            qi = Quaternion(float(Yi[0, i]), Yi[1:4, i].flatten())
            vectorE[:, i] = qi.__mul__(qt_bar.inv()).axis_angle().flatten()

        stuff = np.vstack((vectorE, axisYi[3:] - self.state[4:]))
        for i in range(12):
            cov_xy += stuff[:, i].reshape(-1,1) @ (Zi[:, i] - mean).reshape(1, -1)
        cov_xy = cov_xy / 12

        innovation = observation - mean.reshape(-1, 1)
        # kalman Gain
        K = cov_xy @ np.linalg.inv(cov)
        innK = K @ innovation
        # Calculate new mean and covariance
        self.state[4:] = self.state[4:] + innK[3:]

        innKquat = Quaternion()
        innKquat.from_axis_angle(innK[:3].flatten())
        meanQuat = Quaternion(float(self.state[0]), self.state[1:4].flatten())
        tmp = innKquat.__mul__(meanQuat)
        # tmp.normalize()
        self.state[:4] = (tmp.q).reshape(-1, 1)
        self.filterCov = self.filterCov - K @ cov @ K.T

        self.logger.append(innovation[0])

    def quat2rpy(self, quat):
        r = np.zeros((quat.shape[1],))
        p = np.zeros((quat.shape[1],))
        y = np.zeros((quat.shape[1],))
        for i in range(quat.shape[1]):
            in_q = quat[:, i].flatten()
            q = Quaternion(float(in_q[0]), in_q[1:])
            q.normalize()
            angles = q.euler_angles()
            r[i] = float(angles[0])
            p[i] = float(angles[1])
            y[i] = float(angles[2])
        return r, p, y

    def runPipeline(self):
        # import tqdm
        stateVector = self.state.copy()
        self.logger = []
        nSteps = self.accel.shape[1] - 1
        for i in range(1, nSteps):
            dt = self.timesteps[i] - self.timesteps[i - 1]
            Yi = self.propogateStep(dt)
            observationPacket = np.vstack(
                (self.accel[:, i].reshape(-1, 1), self.gyro[:, i].reshape(-1, 1)))
            self.measurementStep(Yi, observationPacket)
            stateVector = np.column_stack((stateVector, self.state))
        return stateVector


def viconAngle(ang, T):
    return (ang[1:] - ang[:-1]) / 0.01


def estimate_rot(data_num=1):
    # load data
    imu = io.loadmat("imu/imuRaw" + str(data_num) + ".mat")
    accel = imu["vals"][0:3, :]
    gyro = imu["vals"][3:6, :]
    T = (imu["ts"]).flatten()

    sol = ukfPipeline(accel, gyro, T)
    stateVector = sol.runPipeline()


    # roll, pitch, yaw are numpy arrays of length T
    return sol.quat2rpy(stateVector[:4, :])


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    num = 1
    vicon = io.loadmat("vicon/viconRot" + str(num) + ".mat")
    roll, pitch, yaw = estimate_rot(num)

    vicon2Sens = np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]])
    r = []
    p = []
    y = []
    quat = Quaternion()
    for i in range(vicon["rots"].shape[-1]):
        R = vicon["rots"][:, :, i].reshape(3, 3)
        quat.from_rotm(R)
        ang = quat.euler_angles()
        r.append(float(ang[0]))
        p.append(float(ang[1]))
        y.append(float(ang[2]))
    r = np.array(r)
    plt.figure()
    plt.subplot(2, 2, 1)
    plt.plot(r[: roll.shape[0]])
    plt.plot(roll)
    plt.legend(["Vicon","Filtered"])
    plt.title("Roll Angle")
    plt.ylabel("rad")

    plt.subplot(2, 2, 2)
    plt.plot(p[: roll.shape[0]])
    plt.plot(pitch)
    plt.legend(["Vicon","Filtered"])
    plt.title("Pitch Angle")
    plt.ylabel("rad")

    plt.subplot(2, 2, 3)
    plt.plot(y[: roll.shape[0]])
    plt.plot(yaw)
    plt.legend(["Vicon","Filtered"])
    plt.title("Yaw Angle")
    plt.ylabel("rad")

    plt.show()
