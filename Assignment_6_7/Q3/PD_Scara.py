

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve
from controller import PD_Controller,PD_FeedForw_Controller
from traj import oneD_traj

def angle_wrap(angle):
    coeffs = angle%(2*np.pi)
    if abs(coeffs - 2*np.pi)<1e-3:
        coeffs = 0
    return (coeffs)

class SCARA():
    def __init__(self,d1=0.25,a1 = 0.25, a2=0.25):
        self.d1 = d1
        self.a1 = a1
        self.a2 = a2

        self.m1 = 1
        self.m2 = 1
        self.m3 = 1

        self.I1 = 0
        self.I2 = self.m2*self.a1**2/3
        self.I3 = self.m3*self.a2**2/3
        # self.a3 = a3

        self.q =      np.array([0,0,0])
        self.q_dot =  np.array([0,0,0])
        self.q_ddot = np.array([0,0,0])

        # -------------------------------------------
        # self.controller = PD_Controller([[1,0.2],
        #                                  [0.6,0.1],
        #                                  [0,0]])
        # ----------------------------------------------
        self.controller = PD_FeedForw_Controller([[1,0.5],
                                                  [0.6,0.3],
                                                  [1,0]], 20)

        self.isNoise = 0
        self.update_endEffectorPosition()

    def update_endEffectorPosition(self,):
        self.x_EF,self.y_EF,self.z_EF = self.forward_kinematics(self.q)

    def inverse_kinematics(self,points):
        xc,yc,zc = points
        # Workspace condition
        if np.sqrt(xc**2+yc**2)>self.a2 + self.a1:
            print("No Solution can be Found!")
            return False,[None, None, None]
        else: 
            def inv_func(x):
                return [
                        - xc + self.a1*np.cos(x[0]) + self.a2*np.cos(x[0]+x[1]),
                        - yc + self.a1*np.sin(x[0]) + self.a2*np.sin(x[0]+x[1]),
                        - zc + self.d1 - x[2]
                        ]
            root = fsolve(inv_func,[1,1,1])

            q1,q2,d = root

            return True, [angle_wrap(q1),angle_wrap(q2),d]
    
    def forward_kinematics(self,q):
        q1,q2,d = q
        xc = self.a1*np.cos(q1) + self.a2*np.cos(q1+q2)
        yc = self.a1*np.sin(q1) + self.a2*np.sin(q1+q2)
        zc = self.d1 - d

        return xc,yc,zc
    
    def forward_kinematics(self,q):
        q1,q2,d = q
        xc = self.d1*np.cos(q1) + self.a2*np.cos(q1+q2)
        yc = self.d1*np.sin(q1) + self.a2*np.sin(q1+q2)
        zc = self.d1 - d

        return xc,yc,zc
            
            
    def forward_kinematics(self,q):
        q1,q2,d = q
        xc = self.a1*np.cos(q1) + self.a2*np.cos(q1+q2)
        yc = self.a1*np.sin(q1) + self.a2*np.sin(q1+q2)
        zc = self.d1 - d


        return xc,yc,zc

    def dynamics_solver(self,dt,q_des, **kwargs):
        
        g  = 9.8

        e = q_des - self.q

        alpha = self.I1 + self.a1**2*(self.m1/4 + self.m2 + self.m3)
        beta = self.I2 + self.I3 + self.a2**2*(self.m2/4 +self.m3) 
        gamma = self.a1*self.a2*self.m3 + self.a1 * self.a2/2 * self.m2

        self.tau_1,self.tau_2,self.F_3 = self.controller.track_angles(e,qd_dot = kwargs['qd_dot'],qd_ddot = kwargs['qd_ddot'])
        
        if self.isNoise:
            # print(np.random.normal(0,1))
            self.tau_1 +=np.random.normal(0,0.005)
            self.tau_1 +=np.random.normal(0,0.005)
            self.tau_1 +=np.random.normal(0,0.005)

        # Damping term

        u1 = -gamma*np.sin(self.q[1])*self.q_dot[1]*self.q_dot[0] - gamma*np.sin(self.q[1])*(self.q_dot[1] + self.q_dot[0])*self.q_dot[1] + self.tau_1
        u2 = gamma*np.sin(self.q[1])*self.q_dot[0]**2 + self.tau_2
        u3 = self.m3*g + self.F_3

        MM = np.around(np.array([[alpha + beta + 2*gamma*np.cos(self.q[1]), beta + 2*gamma*np.cos(self.q[1]), 0],
                       [beta + 2*gamma*np.cos(self.q[1]), beta, 0],
                       [0, 0, self.m3]]) , decimals= 4)
        
        
        C = np.around(np.array([[-gamma*np.sin(self.q[1])*self.q_dot[1], -gamma*np.sin(self.q[1])*(self.q_dot[1] + self.q_dot[0]), 0],
                      [gamma*np.sin(self.q[1])*self.q_dot[0], 0, 0],
                      [0, 0, 0]]), decimals = 4)
        G = np.around(np.transpose(np.array([0, 0, self.m3*g])), decimals = 4)

        U = np.around(np.transpose(np.array([u1, u2,u3])), decimals = 4)

        K = np.around((U - np.matmul(C, np.transpose(self.q_dot))-G), decimals = 4)

        d2ydt2 = np.around(np.matmul(np.linalg.inv(MM), K), decimals = 4)


        self.q_dot = self.q_dot + d2ydt2*dt
        self.q = self.q + self.q_dot*dt + 1/2*d2ydt2*dt**2
        self.q_ddot = d2ydt2

        for i in range(2):
            self.q[i] = angle_wrap(self.q[i])
        
        self.update_endEffectorPosition()

    def control(self,):
        points, t = oneD_traj([[0.4,0.06,0.1],
                               [0.4,0.01,0.1]],0,10)
        ef = [points[0]]
        des = [points[0]]


        ret, self.q = self.inverse_kinematics(points[0])
        self.q = np.array(self.q)
        self.update_endEffectorPosition()
        
        i = 1
        qd = self.q.copy()
        qd_dot = np.array([0,0,0])

        while True:

            if i<len(t)-1:
                dt = t[i]-t[i-1]
                ret,trk_q = self.inverse_kinematics(points[i])

                qd_dot_ = (trk_q - qd)/dt
                qd_ddot = (qd_dot_ - qd_dot)/dt
                qd_dot = qd_dot_
                qd = np.array(trk_q)


                self.dynamics_solver(dt,trk_q,qd_dot=qd_ddot,qd_ddot=qd_ddot)

                ef.append([self.x_EF,self.y_EF,self.z_EF])
                des.append(points[i])
                # pass
            else:
                break
                pass
            
            i+=1
        ef = np.array(ef)
        des = np.array(des)

        # print(ef)
        plt.figure(0)
        plt.plot(ef[:,1])
        plt.plot(des[:,1])
        plt.grid()

        plt.figure(1)
        plt.plot(ef[:,0])
        plt.plot(des[:,0])
        plt.grid()

        plt.figure(2)
        plt.plot(ef[:,2])
        plt.plot(des[:,2])
        plt.grid()

        plt.show()





            
p = SCARA()
p.control()
# p.find_P2P_trajectory([[0.4,0.06,0.1],
#                        [0.4,0.01,0.1]],
                       
#                        [[0,0,0],
#                         [0,0,0]], 10,'cubic')