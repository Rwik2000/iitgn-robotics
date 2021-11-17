import numpy as np

class PD_Controller():
    def __init__(self,
                 gains=[[1,0],
                        [1,0]],
                ):

        self.gains = np.array(gains)
        self.error_buffer = np.zeros((len(gains),5))

    def sum_errors(self):
        error_sum = []
    
        for i in range(len(self.error_buffer)):
            error_sum.append(np.sum(self.error_buffer))

        # print(np.array(error_sum))
        return np.array(error_sum)

    def track_angles(self,errors,**kwargs):
        
        self.error_buffer = np.hstack((self.error_buffer[:,1:],np.transpose([errors])))
        
        prop = self.gains[:,0]*errors
        deri = self.gains[:,1]*(self.error_buffer[:,-1] - self.error_buffer[:,-2])

        # No derivative term
        outputs = prop + deri
        return outputs

class PD_FeedForw_Controller():
    def __init__(self,
                 gains=[[1,0],
                        [1,0]],
                 K_motor = 0.1
                ):

        self.J_eff = 1
        self.B_eff = 1
        self.K_motor = K_motor
        self.gains = np.array(gains)
        self.error_buffer = np.zeros((len(gains),5))

    def sum_errors(self):
        error_sum = []
    
        for i in range(len(self.error_buffer)):
            error_sum.append(np.sum(self.error_buffer))

        # print(np.array(error_sum))
        return np.array(error_sum)

    def track_angles(self,errors,**kwargs):
        qd_dot = kwargs['qd_dot']
        qd_ddot = kwargs['qd_ddot']
        
        self.error_buffer = np.hstack((self.error_buffer[:,1:],np.transpose([errors])))
        
        prop = self.gains[:,0]*errors
        deri = self.gains[:,1]*(self.error_buffer[:,-1] - self.error_buffer[:,-2])

        feed_forward = (self.J_eff*qd_ddot + self.B_eff*qd_dot)/self.K_motor
        # No derivative term
        outputs = prop + deri + feed_forward
        return outputs

class PD_FF_NoiseCancel_Controller():
    def __init__(self,
                 gains=[[1,0],
                        [1,0]],
                 K_motor = 0.1
                ):

        self.J_eff = 1
        self.B_eff = 1
        self.K_motor = K_motor
        self.gains = np.array(gains)
        self.error_buffer = np.zeros((len(gains),5))

    def sum_errors(self):
        error_sum = []
    
        for i in range(len(self.error_buffer)):
            error_sum.append(np.sum(self.error_buffer))

        # print(np.array(error_sum))
        return np.array(error_sum)

    def track_angles(self,errors,**kwargs):
        qd_dot = kwargs['qd_dot']
        qd_ddot = kwargs['qd_ddot']
        
        noise = np.random.normal(0,1,100)
        
        self.error_buffer = np.hstack((self.error_buffer[:,1:],np.transpose([errors])))
        
        prop = self.gains[:,0]*errors
        deri = self.gains[:,1]*(self.error_buffer[:,-1] - self.error_buffer[:,-2])

        feed_forward = (self.J_eff*qd_ddot + self.B_eff*qd_dot)/self.K_motor
        # No derivative term
        outputs = prop + deri + feed_forward
        return outputs


