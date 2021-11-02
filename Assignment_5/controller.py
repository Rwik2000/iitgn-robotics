import numpy as np
from scipy.spatial.transform import Rotation as R

class PI_Controller():
    def __init__(self,
                 gains=[[1,3],
                        [1,3]],
                ):

        self.gains = np.array(gains)
        self.error_buffer = np.zeros((len(gains),10))

    def sum_errors(self):
        error_sum = []
    
        for i in range(len(self.error_buffer)):
            error_sum.append(np.sum(self.error_buffer))

        return np.array(error_sum)

    def track_angles(self,errors):
        
        self.error_buffer = np.hstack((self.error_buffer[:,1:],np.transpose([errors])))
        
        prop = self.gains[:,0]*errors
        intgr = self.gains[:,1]*self.sum_errors()
        
        # No derivative term
        outputs = prop + intgr
        return outputs

