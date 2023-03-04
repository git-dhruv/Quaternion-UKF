# #simulator!!


import matplotlib.pyplot as plt
import numpy as np


class q1():
    def __init__(self):
        self.a = -1
        self.initX = np.random.normal(1,2)
        
        self.observations = []
    def simulateODE(self, processNoise, x, a = None):
        #Simulates a simple equation which is not even an ODE
        if a is None:
            return self.a*x + processNoise
        else:
            return a*x 
    def generateNoise(self,nRuns):
        self.processNoise = np.random.normal(0,1,size=(nRuns,))
        self.observationNoise = np.random.normal(0,np.sqrt(1/2),size=(nRuns,))

    def genGroundTruth(self,nRuns):
        """
        Q1(a)
        """
        self.observations = []
        self.path = []
        self.generateNoise(nRuns)
        x = self.initX
        for i in range(nRuns):
            x = self.simulateODE(self.processNoise[i],x)
            y = np.sqrt(x*x+1) + self.observationNoise[i]
            self.observations.append(y)
            self.path.append(x)

    def plot(self):
        plt.plot(self.path)
        plt.plot(self.observations,'--')
        plt.legend(["State (mean)","Observations"])
        plt.xlabel("No. of Iterations")
        plt.show()

    def runPipeline(self, nRuns):
        plotx = 0



        q1_data = []
        q2_data = []

        q = np.array([self.initX,-20])
        A = np.array([[q[0],q[1]],[0,1]])   
        C = np.array([[q[0]/np.sqrt(q[0]*q[0] + 1), 0 ],[0, 0]])
        Q = 0.5
        R = np.diag([1,.34])

        Covariance = np.diag([10,10])
        for i in range(nRuns):
            #Propogation/Prediction
            A = np.array([[q[1],q[0]],[0,1]], dtype=np.float64)

            q = np.array([[self.simulateODE(0,float(q[0]), a = float(q[1]))],
                            [q[1]]], dtype=np.float64)
            
            Covariance = A@Covariance@A.T + R
            q = q.flatten()

            #Kalman Gain
            C = np.array([[float(q[0])/np.sqrt(float(q[0]*q[0]) + 1), 0 ]], dtype=np.float64)
            K = Covariance @ C.T @ np.linalg.pinv(C@Covariance@C.T + Q)

            #Observation
            q = q.reshape(-1,1) + K*np.array([self.observations[i] - np.sqrt(q[0]*q[0]+1)],dtype=np.float64).reshape(-1,1)
            Covariance = (np.eye(2) - K@C)@Covariance
            q = q.flatten()
            if plotx==0:
                q1_data.append(np.sqrt(Covariance[-1,-1]))
                q2_data.append(q[1])
            else:
                q1_data.append(np.sqrt(Covariance[0,0]))
                q2_data.append(q[0])


        """
        before you run this, for plotting both covariance and a, u have to change plotX in the first line of this function
        """
        if plotx==0:
            covs = np.sqrt(np.array(q1_data)).reshape(-1,1)
            plt.plot(q2_data)
            plt.plot(-1*np.ones((100,1)),'--')
            plt.plot(q2_data+covs.flatten(), 'k.')
            plt.plot(q2_data-covs.flatten(), 'k.')
            plt.legend(["a","True Value","sigma"])
            plt.show()
        else:
            covs = np.sqrt(np.array(q1_data)).reshape(-1,1)
            plt.plot(q2_data)
            plt.plot(self.path)
            plt.plot(self.path+covs.flatten(), 'k.')
            plt.plot(self.path-covs.flatten(), 'k.')
            plt.legend(["x","True Value","sigma"])
            plt.show()


            





ans = q1()
ans.genGroundTruth(100)
# ans.plot()  #This is for q1(a)
ans.runPipeline(100)   #This is for q2(a), run multiple times


