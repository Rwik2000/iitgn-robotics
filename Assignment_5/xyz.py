import numpy as np
from scipy.optimize import fsolve
from scipy.spatial.transform import Rotation 
from controller import PI_Controller

def angle_wrap(angle):
    a = angle%(2*np.pi)
    if abs(a - 2*np.pi)<1e-3:
        a = 0
    return (a)

class manipulator():
    def __init__(self,dh_params, config=['R','R']):
        # configuration of the manipulator. User has 2 choices "R"->revolute. "P"->prismatic.
        # Default configuration is a 2R manipulator with all the angles at 0 degrees and lengths being 1 unit.
        self.config = config 
        # User must input the dh parameters in matrix form i.e. "R"->revolute
        # [[a1 , alpha1 , d1, theta1]
        #  [a2 , alpha2 , d2, theta2]
        #  .
        #  .
        #  .
        #  [an , alphan , dn, thetan]]
        # n being the nth link of the manipulator.
        self.dh=dh_params
    
    def calc_tranfMatrix(self, dh_params,i):
        # Calculating Trnasformation matrix
        a, alpha,d,theta = dh_params
        A = np.array([[np.cos(theta), -np.sin(theta)*np.cos(alpha),  np.sin(theta)*np.sin(alpha), a*np.cos(theta)],
                      [np.sin(theta),  np.cos(theta)*np.cos(alpha), -np.cos(theta)*np.sin(alpha), a*np.sin(theta)],
                      [            0,                np.sin(alpha),                np.cos(alpha),               d],
                      [            0,                            0,                            0,               1]])
        return A 

    def forward_kinematics(self,):
        tr=self.calc_tranfMatrix(self.dh[0],0)
        Trs = [tr]
        # Calculating the individual transformation matrices. And appending to the T matrix in the following form.
        # A1
        # A1A2
        # A1A2A3 ... 
        for i in range(len(self.dh)-1):
            tr = np.matmul(tr,self.calc_tranfMatrix(self.dh[i+1],i+1))
            Trs.append(tr)

        # Calculating the jacobian matrix
        h = []
        for i in range(len(self.config)):
            temp2 = np.array([0,0,0])
            if self.config[i]=='R':
                temp2 = np.array([0,0,1])
            if  i ==0:
                temp = np.array(Trs[-1])

            else:
                temp = np.array(Trs[-1]) - np.array(Trs[i-1])
            
            h.append(np.cross(temp2,temp[:3,3:].transpose()).transpose())
        
        # Velocity jacobian
        J_v = h[0]
        for i in range(len(self.config)-1):
            J_v = np.hstack((J_v,h[i+1]))
        
        
        # Angular velocity jacobian
        J_omega = np.array([[0],[0],[1]])
        if self.config[0]=='P':
            J_omega = np.array([[0],[0],[0]])

        for i in range(len(self.config)-1):
            temp = np.array([[0],[0],[1]])

            if self.config[i+1]=='P':
                temp = np.array([[0],[0],[0]])
            J_omega = np.hstack((J_omega,temp))
        
        # Overall Transformation matrix T06.
        transformation_matrix = np.array(Trs[-1])
        # Manipulator jacobian
        J = np.vstack((J_v,J_omega))

        return {'transformation_matrix':transformation_matrix,'jacobian':J}

class PUMA():
    def __init__(self,d1=1,a2=1,a3=1):
        self.d1 = d1
        self.a2 = a2
        self.a3 = a3
        self.q1 = 0
        self.q2 = 0
        self.q3 = 0

        self.update_endEffectorPosition()

    def update_endEffectorPosition(self,):
        self.xE,self.yE,self.zE = self.forward_kinematics(self.q1,self.q2,self.q3)
    
    def inverse_kinematics(self,xc,yc,zc):

        # Workspace condition
        D = (xc**2 + yc**2 + (zc-self.d1)**2 - self.a2**2 - self.a3**2)/(2*self.a2*self.a3)
        if abs(D)<=1:
            def inv_func(x):
                return [
                        -xc + self.a2*np.cos(x[1])*np.cos(x[0]) + self.a3*np.cos(x[1]+x[2])*np.cos(x[0]),
                        -yc + self.a2*np.cos(x[1])*np.sin(x[0]) + self.a3*np.cos(x[1]+x[2])*np.sin(x[0]),
                        -zc + self.d1 + self.a2*np.sin(x[1]) + self.a3*np.sin(x[1]+x[2])
                        ]
            root = fsolve(inv_func,[0,0,0])
            q1,q2,q3 = root
            
            self.q1,self.q2,self.q3 = angle_wrap(q1),angle_wrap(q2),angle_wrap(q3)
            # Returns True if solution exists
            return True, [self.q1,self.q2,self.q3]
        else:
            # Returns True if solution exists
            print("Angles provided not in workspace")
            return False, [None, None, None]
    
    def forward_kinematics(self,q1,q2,q3):
        xc = self.a2*np.cos(q2)*np.cos(q1) + self.a3*np.cos(q2+q3)*np.cos(q1)
        yc = self.a2*np.cos(q2)*np.sin(q1) + self.a3*np.cos(q2+q3)*np.sin(q1)
        zc = self.d1 + self.a2*np.sin(q2) + self.a3*np.sin(q2+q3)

        self.xE,self.yE,self.zE = xc,yc,zc

        return xc,yc,zc
    
    def point_tracking(self,trk_points):
        # dh = 
        # TMatrix = 
        xt,yt,zt = trk_points

        ret, angles = self.inverse_kinematics(xt,yt,zt)
        if ret:
            print(angles) 
        else:
            print('Error Occurred! Exiting ...')


robot = PUMA()
robot.point_tracking([0,2,1])

