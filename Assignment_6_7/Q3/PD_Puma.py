

import numpy as np
import matplotlib.pyplot as plt
from traj import oneD_traj
from controller import PD_Controller, PD_FeedForw_MotorController
from xyz import findDH_manipulator

def angle_wrap(angle):
    coeffs = angle%(2*np.pi)
    if abs(coeffs - 2*np.pi)<1e-3:
        coeffs = 0
    return (coeffs)

class PUMA():
    def __init__(self,d1=0.25,a2=0.25,a3=0.25):
        self.d1 = d1
        self.a2 = a2
        self.a3 = a3
        self.m1 = 1
        self.m2 = 1
        self.m3 = 1

        self.I1 = self.m1*self.d1**2/3
        self.I2 = self.m2*self.a2**2/3
        self.I3 = 0


        self.q =      np.array([0,0,0])
        self.q_dot =  np.array([0,0,0])
        self.q_ddot = np.array([0,0,0])

        # -------------------------------------------
        #
        # self.controller = PD_Controller([[1,0],
        #                                  [1,0],
        #                                  [1,0]])
        # -------------------------------------------
        # 
        self.controller = PD_FeedForw_MotorController([[1,0],
                                                       [1,0],
                                                       [1,0]], 0.1)

        self.dh_manipulator = findDH_manipulator(['R','R','R'])
        self.update_endEffectorPosition()
    
    def update_endEffectorPosition(self,):
        self.x_EF,self.y_EF,self.z_EF = self.forward_kinematics(self.q)

    def inverse_kinematics(self,point):
        
        q1 = np.arctan2(point[1],point[0])
        D = (point[0]**2 + point[1]**2 + (point[2]-self.d1)**2 -self.a2**2 - self.a3**2)/(2*self.a2*self.a3)
        if abs(D)<=1:
            q3 = np.arctan2(np.sqrt(1-D**2),D)
            q2 = np.arctan2(point[2]-self.d1,np.sqrt(point[0]**2 + point[1]**2)) - np.arctan2(self.a2*np.sin(q3),self.a2 + self.a3*np.cos(q3))

            return True, [angle_wrap(q1),angle_wrap(q2),angle_wrap(q3)]

        else:
            print("Error. The given inputs are out of bounds of workspace")
            return False, [None,None,None]
            
            
    def forward_kinematics(self,q):
        xc = self.a2*np.cos(q[1])*np.cos(q[0]) + self.a3*np.cos(q[1]+q[2])*np.cos(q[0])
        yc = self.a2*np.cos(q[1])*np.sin(q[0]) + self.a3*np.cos(q[1]+q[2])*np.sin(q[0])
        zc = self.d1 + self.a2*np.sin(q[1]) + self.a3*np.sin(q[1]+q[2])

        return xc,yc,zc

    def dynamics_solver(self,dt,x_des):
        
        g  = 9.8

        e = x_des - self.x

        alpha = self.J1 + self.a1**2*(self.m1/4 + self.m2 + self.m3)
        beta = self.J2 + self.J3 + self.a2**2*(self.m2/4 +self.m3) 
        gamma = self.a1*self.a2*self.m3 + self.a1 * self.a2/2 * self.m2

        self.tau_1,self.tau_2,self.F_3 = self.controller.track_angles(e)

        # Damping term

        u1 = -gamma*np.sin(self.x[1])*self.x_dot[1]*self.x_dot[0] - gamma*np.sin(self.x[1])*(self.x_dot[1] + self.x_dot[0])*self.x_dot[1] + self.tau_1
        u2 = gamma*np.sin(self.x[1])*self.x_dot[0]**2 + self.tau_2
        u3 = self.m3*g + self.F_3

        MM = np.around(np.array([[alpha + beta + 2*gamma*np.cos(self.x[1]), beta + 2*gamma*np.cos(self.x[1]), 0],
                       [beta + 2*gamma*np.cos(self.x[1]), beta, 0],
                       [0, 0, self.m3]]) , decimals= 4)
        
        
        C = np.around(np.array([[-gamma*np.sin(self.x[1])*self.x_dot[1], -gamma*np.sin(self.x[1])*(self.x_dot[1] + self.x_dot[0]), 0],
                      [gamma*np.sin(self.x[1])*self.x_dot[0], 0, 0],
                      [0, 0, 0]]), decimals = 4)
        G = np.around(np.transpose(np.array([0, 0, self.m3*g])), decimals = 4)

        U = np.around(np.transpose(np.array([u1, u2,u3])), decimals = 4)

        K = np.around((U - np.matmul(C, np.transpose(self.x_dot))-G), decimals = 4)

        d2ydt2 = np.around(np.matmul(np.linalg.inv(MM), K), decimals = 4)


        self.x_dot = self.x_dot + d2ydt2*dt
        self.x = self.x_dot*dt + 1/2*d2ydt2*dt**2
        for i in range(2):
            self.x[i] = angle_wrap(self.x[i])
        
        self.update_endEffectorPosition()

    def control(self,):
        points, t = oneD_traj([[0.4,0.06,0.1],
                               [0.4,0.01,0.1]],0,10)
        
        q1,q2,q3 = self.q
        dh_params = [[0,np.pi/2,    1,q1],
                     [1,      0,    0,q2],
                     [1,      0,    0,q3]]

        a = self.dh_manipulator.forward(dh_params)
        print(a['trans_mtrx'][-1])
        
        pass

    




            
p = PUMA()
p.control()