import numpy as np

class collision:
    def __init__(self):
        pass

    def line_plane_collision(self,P1,P2,n):
        n_2d = np.tile(n, (P1.shape[0],1))
        P1_n = 