#--------------------------------------------------------------------
# Implements 2 DoF Robotic Arm Training anf testing to play with
# a ball using KNN Regression
#
# Author: Sriram Emarose [sriram.emarose@gmail.com]
#
#--------------------------------------------------------------------

import math
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D
from sklearn.neighbors import KNeighborsRegressor
import random, time, csv

# Global Constants
l1 = 10 # Length of link1
l2 = 10 # Length of link2
robotOrigin = [0,0]
X = 0
Y = 1
ROBOT_RUN_ITERATION = 10000
GRIPPER_OFFSET = 5
TRAIN_DATA_FILE = 'ballShooterTraining.txt'

# Robot running modes
TRAIN = 0
TEST = 1

class RobotKnnRegressor:

    def Train(self):
        train = pd.read_csv(TRAIN_DATA_FILE)
        cols = ['theta1', 'theta2']
        labelCol = ['groundTruthX', 'groundTruthY']

        theta_GroundTruth = train.as_matrix(cols)
        ballHit_GroundTruth = train.as_matrix(labelCol)

        knn = KNeighborsRegressor(n_neighbors=1)

        print('KNN Training under Progress..Please wait..')
        knn.fit(ballHit_GroundTruth, theta_GroundTruth)

        print('Training Complted..')
        predicted = knn.predict(ballHit_GroundTruth)

        '''
        predictedX=np.array(predicted[:,0], np.int32)
        predictedY = np.array(predicted[:, 1], np.int32)
        plt.axes(xlim=(-50, 50), ylim=(-50, 50))
        plt.autoscale(True)

        gndTx = np.array(theta_GroundTruth[:,0], np.int32)
        gndTy = np.array(theta_GroundTruth[:, 1], np.int32)
        plt.scatter(gndTx, gndTy, edgecolors='blue')
        plt.scatter(predictedX, predictedY, edgecolors='red', marker=">", s=30)
        plt.show()
        '''
        return knn


    def SimulateRobot(self, mode):
        ballCoord = GetNewBall()
        gotTrainedModel = False
        knnTrained = None
        hitCount = 0
        lostCount = 0
        ballHit = False
        predictedCoords = [[10, 10],[45, 45]]

        for i in range(ROBOT_RUN_ITERATION):

            if (ballCoord[1] < 5):
                # Ball went beyond robot's reach. Retry!
                ballCoord = GetNewBall()
                #if ballHit == False:
                lostCount += 1
            else:
                ballCoord[1] -= 1

            if mode == TRAIN:
                randTheta1 = math.radians(random.randrange(0, 180)) # should range from 0 to 90(PI/2)
                randTheta2 = math.radians(random.randrange(0, 120)) # Should range from 0 to 180(PI)

                coords = ComputeEndEffectorPos(randTheta1, randTheta2)

                PlotSimulation(coords, ballCoord, '-rx', '-bo',0,0)

                # if ball is hit, then save the context of gripper
                # this is our positive data
                xMinRange = coords[X][1] - GRIPPER_OFFSET
                xMaxRange = coords[X][1] + GRIPPER_OFFSET

                yMinRange = coords[Y][1] - GRIPPER_OFFSET
                yMaxRange = coords[Y][1] + GRIPPER_OFFSET

                if (ballCoord[X] >= xMinRange or ballCoord[X] <= xMaxRange) and \
                        (ballCoord[Y] >= yMinRange or ballCoord[Y] <= yMaxRange):
                    WriteTrainingData([randTheta1, randTheta2, ballCoord[0], ballCoord[1]])
            else:
                if ballCoord[Y] < 20:
                    # Prediction Mode
                    if gotTrainedModel == False:
                        knnTrained = self.Train()
                        gotTrainedModel = True

                    predictedRobottheta = knnTrained.predict([[ballCoord[0], ballCoord[1]]])

                    predictedT1 = np.array(predictedRobottheta[:, 0], np.int32)
                    predictedT2 = np.array(predictedRobottheta[:, 1], np.int32)
                    predictedCoords = ComputeEndEffectorPos(predictedT1, predictedT2)

                    xMinRange = predictedCoords[X][1] - GRIPPER_OFFSET
                    xMaxRange = predictedCoords[X][1] + GRIPPER_OFFSET

                    yMinRange = predictedCoords[Y][1] - GRIPPER_OFFSET
                    yMaxRange = predictedCoords[Y][1] + GRIPPER_OFFSET

                    if (ballCoord[X] >= xMinRange and ballCoord[X] <= xMaxRange) and \
                            (ballCoord[Y] >= yMinRange and ballCoord[Y] <= yMaxRange):
                        PlotSimulation(predictedCoords, ballCoord, '-go', '-ro', hitCount, lostCount)
                        ballCoord = GetNewBall()
                        hitCount += 1
                else:
                    PlotSimulation(predictedCoords, ballCoord, '-ro', '-bo', hitCount, lostCount)


def LoadTrainingData():
   theta1Vec = []
   theta2Vec = []
   groundTruth_ballHitX  = []
   groundTruth_ballHitY  = []

   with open(TRAIN_DATA_FILE, 'r') as trainedFile:
       row = csv.reader(trainedFile, delimiter=',')

       for val in row:
           theta1Vec.append(int(val[0]))
           theta2Vec.append(int(val[1]))
           groundTruth_ballHitX.append(int(val[2]))
           groundTruth_ballHitY.append(int(val[3]))

   return [theta1Vec, theta2Vec, groundTruth_ballHitX, groundTruth_ballHitY]



def WriteTrainingData(data):
    file = open(TRAIN_DATA_FILE, 'a')

    for i in range(len(data)):
        val = '%d' %data[i]
        file.write(val)
        file.write(',')
    file.write('\n')
    file.close()


# Calculates coordinate(x,y) of a given theta1 and theta2
# using forward kinematics
def ComputeEndEffectorPos(theta1, theta2):
    link1_x = (l1 * math.cos(theta1))
    link1_y = (l1 * math.sin(theta1))
    link2_x = link1_x + (l2 * math.cos(theta1 + theta2))
    link2_y = link1_y + (l2 * math.sin(theta1 + theta2))
    return (([link1_x, link2_x], [link1_y,link2_y]))

def GetNewBall():
    ballX = random.randrange(-20, 20)
    ballY = random.randrange(25, 50)
    return [ballX, ballY]

def PlotSimulation(coords, ballCoord, ballMarker, robotMarker, hitCount, lostCount):
    plt.ion()
    plt.axes(xlim=(-50, 50), ylim=(-50, 50))
    '''
    plt.gca(projection='3d')
    axes = plt.gca()
    axes.set_xlim([-20, 20])
    axes.set_ylim([-10, 10])
    axes.set_zlim([-10, 10])
    '''
    plt.axis('off')
    plt.autoscale(False)
    plt.annotate('Hits = ', xy=(-45,40))
    plt.annotate(str(hitCount), xy=(-30, 40))
    plt.annotate('Lost = ', xy=(-45,35))
    plt.annotate(str(lostCount), xy=(-30, 35))

    plt.plot([robotOrigin[X], coords[X][0], coords[X][1]],
             [robotOrigin[Y], coords[Y][0], coords[Y][1]], robotMarker)
    plt.plot([ballCoord[0]], [ballCoord[1]], ballMarker)
    plt.scatter(coords[X][1], coords[Y][1], color='pink', marker="o", s=500)

    plt.plot([-50,50],[-1,-1], 'k')
    plt.pause(0.05)
    plt.cla()


def main():

    knnTrainer = RobotKnnRegressor()
    knnTrainer.SimulateRobot(TEST)
    #knnTrainer.SimulateRobot(TRAIN)





if __name__ == "__main__":
    main()
