import numpy as np
import csv
import time
from matplotlib import pyplot as plt

class calculate:
    def __init__(self,under_theta,top_theta,angle_step_size):
        # classの初期値定義
        self.len_values = np.array([15,12,24,9.5,21,11,8.5,8.1])
        self.dh_size = 9
        self.th_block = 7
        # 距離が閾値以内の行のインデックスを格納
        self.indices = []
        self.shin_th_list = [None] * self.th_block
        self.position = np.zeros(15)
        self.indnum_2 = 0

        # colon演算子を使用して配列を作成
        rangeArray = np.arange(under_theta, top_theta, angle_step_size)
        self.inputCell = [rangeArray]*5
        # 各配列からなるグリッドを生成
        grid2 = np.meshgrid(*self.inputCell)
        # グリッドから直積を生成
        th_list = np.array([grid.flatten() for grid in grid2]).T
        self.th_list = th_list[np.argsort(th_list[:, 0])]
        block_size = th_list.shape[0] // self.th_block
        for i in range(self.th_block):
            start_index = i * block_size
            # For the last block, capture remaining elements as well
            end_index = (i + 1) * block_size if i != (self.th_block - 1) else th_list.shape[0]
            self.shin_th_list[i] = th_list[start_index:end_index]

    
    # Convert degrees to radians
    def deg2rad(self, deg):
        return deg * np.pi / 180
    
    def create_transformation_matrix(self, a, alpha, d, theta):
        transformation_matrix = np.array([[np.cos(theta), -np.sin(theta), 0, a],
                                          [np.cos(alpha)*np.sin(theta), np.cos(alpha)*np.cos(theta), -np.sin(alpha), -d*np.sin(alpha)],
                                          [np.sin(alpha)*np.sin(theta), np.sin(alpha)*np.cos(theta), np.cos(alpha), d*np.cos(alpha)],
                                          [0, 0, 0, 1]])
        return transformation_matrix
    
    
    def forward_kinematics_once(self, k):
        # Initialize theta
        theta_list = self.shin_th_list[k]
        self.theta_list_size = theta_list.shape[0]
        len = self.len_values
        sum_dh_para = np.zeros((self.theta_list_size,self.dh_size,4))
        start_time = time.time()  # 開始時間を記録
        for i in range(self.theta_list_size):
            theta = theta_list[i]
            dh_para = np.array([[0 ,0 ,0 ,0],
                                [0, 0, len[0], theta[0]],
                                [0, 90, len[1], 90+theta[1]],
                                [len[2], 0, 0, 0],
                                [0, 180, len[3], theta[2]],
                                [len[4],0,0,0],
                                [0, 180, len[5], -90+theta[3]],
                                [0, -90, len[6], theta[4]],
                                [0, 90, len[7], 90],
                                ])
            sum_dh_para[i] = dh_para
        end_time = time.time()
        print(f"makedh処理時間: {end_time - start_time}秒")            
        # Convert degrees to radians
        sum_dh_para[:, :, [1, 3]] = self.deg2rad(sum_dh_para[:, :, [1, 3]])
        start_time = time.time()  # 開始時間を記録
        # Calculate transformation matrices
        T_all = [[self.create_transformation_matrix(*sum_dh_para[j][i]) for i in range(sum_dh_para.shape[1])] for j in range(sum_dh_para.shape[0])]
        end_time = time.time()
        print(f"create_trans処理時間: {end_time - start_time}秒")
        # Calculate world coordinates
        start_time = time.time()  # 開始時間を記録
        x, y, z = np.zeros((self.theta_list_size, self.dh_size)), np.zeros((self.theta_list_size, self.dh_size)), np.zeros((self.theta_list_size, self.dh_size))
        for j in range(self.theta_list_size):
            T_all_comp = np.eye(4)
            for i in range(self.dh_size):
                T_all_comp = np.matmul(T_all_comp, T_all[j][i])
                x[j][i], y[j][i], z[j][i] = T_all_comp[:3, 3]
        end_time = time.time()
        print(f"link_point処理時間: {end_time - start_time}秒")
        return x , y , z

    
    
    def create_C_table_once(self, j):
        #theta群の作成
        theta_list = self.shin_th_list[j]
        #C_tableの初期化
        c_space_vector = np.zeros((11,theta_list.shape[0]))
        #始点と終点が保存されたtableにthetaを付け加えるため，axis=0方向へと9倍拡張
        theta_list_3d = np.tile(theta_list,( self.dh_size ,1,1))
        #axis=0とaxis=2の軸を入れ替え
        theta_list_3d = theta_list_3d.transpose()
        #DHパラメータを用いた順運動学により関節のxyz座標を取得
        x , y , z  = self.forward_kinematics_once(j)
        #xyz座標を一つの行列へ(3,:,9)
        vector = np.stack([x,y,z])
        #axis=2方向へ一つ要素をずらす
        vector_roll = np.roll(vector, -1 , axis=2)
        #始点と終点が格納された行列の作成(6,:,9)
        rob_vector = np.concatenate([vector,vector_roll],0)
        #C_tableの作成(6+5,:,9)
        c_space_vector_3d = np.concatenate([rob_vector , theta_list_3d] , 0)
        #axis=2の9番目は最後の関節点から最初の関節点が保存されているため削除(6+5,:,8)
        c_space_vector_3d = np.delete(c_space_vector_3d , self.dh_size-1 , 2 )
        c_space_vector = c_space_vector_3d[:,:,0]
        for i in range(1,self.dh_size-1):
            c_space_vector = np.concatenate([c_space_vector,c_space_vector_3d[:,:,i]],1)
        c_space_vector = c_space_vector.transpose()
        return c_space_vector

    

    def create_grid_table(self):
        for i in range(7):
            start_time = time.time()  # 開始時間を記録
            C_table = self.create_C_table_once(i) 
            end_time = time.time()
            print(f"c_table時間: {end_time - start_time}秒")
            np.savetxt('grid/fake5_C' + str(i) + '.txt', C_table )


def plot_3d(x, y, z):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x, y, z, marker='o')
    ax.plot(x, y, z, '-o') # '-o' で点をプロットし、その点を線で結びます。
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
    max_range = np.array([x.max()-x.min(), y.max()-y.min(), z.max()-z.min()]).max() * 0.5

    mid_x = (x.max()+x.min()) * 0.5
    mid_y = (y.max()+y.min()) * 0.5
    mid_z = (z.max()+z.min()) * 0.5
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)
    plt.show()

def new_plot_3d(x, y, z, xx,yy,zz):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x, y, z, marker='o')
    ax.scatter(xx,yy,zz, c='red')
    ax.plot(x, y, z, '-o') # '-o' で点をプロットし、その点を線で結びます。
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
    max_range = np.array([x.max()-x.min(), y.max()-y.min(), z.max()-z.min()]).max() * 0.5

    mid_x = (x.max()+x.min()) * 0.5
    mid_y = (y.max()+y.min()) * 0.5
    mid_z = (z.max()+z.min()) * 0.5
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)
    plt.show()

def test():
    test_c_space = calculate(-90,91,15)
    x,y,z = test_c_space.forward_kinematics_once(5)
    theta=test_c_space.shin_th_list[5]

    for i in range(x.shape[0]):
        print(theta[i])
        plot_3d(x[i], y[i], z[i])
        
def test2():
    test_c_space = calculate(-90,91,5)
    vec = test_c_space.create_C_table_once(5)
    





def main():
    c_space_maker = calculate(-90,91,5)
    c_space_maker.create_grid_table()
if __name__=="__main__":
    main()
    # test2()