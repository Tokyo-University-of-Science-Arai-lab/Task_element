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
        # 閾値を設定
        self.threshold = 4
        # 距離が閾値以内の行のインデックスを格納
        self.indices = []
        self.shin_th_list = [None] * self.th_block
        self.position = np.zeros(15)
        self.indnum_2 = 0
        # 格子点の範囲と間隔を定義
        self.x_gr = np.arange(-50, 51, 2)
        self.y_gr = np.arange(0, 51, 2)
        self.z_gr = np.arange(0, 101, 2)
        # numpyのmeshgrid関数で全ての組み合わせを生成
        X, Y, Z = np.meshgrid(self.x_gr, self.y_gr, self.z_gr)
        # 結果を1つの配列にリシェイプ
        grid_sp = np.array([X.flatten(), Y.flatten(), Z.flatten()])
        self.grid_space = grid_sp.T
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
        # link_position
        # self.x = np.zeros(self.len_size)
        # self.y = np.zeros(self.len_size)
        # self.y = np.zeros(self.len_size)
    
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

    def find_gird(self , j):
        theta_list = self.shin_th_list[j]
        x , y , z  = self.forward_kinematics_once(j)
        grid_table =[]
        print(self.theta_list_size)
        for l in range(self.theta_list_size):
            start_time = time.time()  # 開始時間を記録
            for m in range(self.dh_size-1):     # 関節点を結ぶ直線を考える
                p1 = np.array([x[l][m], y[l][m], z[l][m]])    # 始点
                p2 = np.array([x[l][m+1], y[l][m+1], z[l][m+1]])    # 終点
                v = p2 -p1 # ベクトルp1からp2へのベクトルを計算します
                w = self.grid_space - p1  # 各格子点から始点までのベクトルを計算します
                t = np.dot(w, v) / np.dot(v, v)  # 各格子点から直線に下ろした垂線の足を求めます
                p3 = p1 + t[:, np.newaxis] * v
                t_min = 0  # p1に対応
                t_max = 1  # p2に対応
                in_range = (t >= t_min) & (t <= t_max)  # 点p1とp2に対応するtの範囲内にある点だけを取り出します
                in_range_grid_space = self.grid_space[in_range]  # tの範囲内の行を取得します
                distances = np.sqrt(np.sum((in_range_grid_space - p3[in_range]) ** 2, axis=1))  # 各格子点から垂線の足までの距離を計算します
                within_threshold_grid_space = in_range_grid_space[distances <= self.threshold]  # 距離が閾値以内の行を取得します

                w2 = within_threshold_grid_space - p1  # 各格子点から始点までのベクトルを計算します
                t2 = np.dot(w2, v) / np.dot(v, v)  # 各格子点から直線に下ろした垂線の足を求めます
                p3_2 = p1 + t2[:, np.newaxis] * v
                in_range2 = (t2 >= t_min) & (t2 <= t_max)
                distances2 = np.sqrt(np.sum((within_threshold_grid_space - p3_2[in_range2]) ** 2, axis=1))

                within_threshold_grid_space = within_threshold_grid_space[self.threshold-2 <= distances2] 
                within_threshold_grid_space_size= within_threshold_grid_space.shape[0]  # 格子点の量の取得
                grid_space_once = np.zeros((within_threshold_grid_space_size , 8))  # 初期値の作成
                for item in range(within_threshold_grid_space_size):
                    grid_space_once[item] = np.append(within_threshold_grid_space[item], theta_list[l])
                grid_table.extend(grid_space_once.tolist())
            end_time = time.time()
            print(f"処理時間: {end_time - start_time}秒")
            
            return grid_table
    
    def create_C_table_once(self, j):
        theta_list = self.shin_th_list[j]
        theta_list_3d = np.tile(theta_list,( self.dh_size ,1,1))
        theta_list_3d = theta_list_3d.transpose()
        x , y , z  = self.forward_kinematics_once(j)
        vector = np.stack([x,y,z])
        vector_roll = np.roll(vector, -1 , axis=2)
        rob_vector = np.concatenate([vector,vector_roll],0)
        c_space_vector = np.concatenate([rob_vector , theta_list_3d] , 0)
        return c_space_vector

    def find_gird_test(self , j , l):
        theta_list = self.shin_th_list[j]
        x , y , z  = self.forward_kinematics_once(j)
        grid_table =[]
        print(self.theta_list_size)

        start_time = time.time()  # 開始時間を記録
        for m in range(self.dh_size-1):     # 関節点を結ぶ直線を考える
            p1 = np.array([x[l][m], y[l][m], z[l][m]])    # 始点
            p2 = np.array([x[l][m+1], y[l][m+1], z[l][m+1]])    # 終点
            v = p2 -p1 # ベクトルp1からp2へのベクトルを計算します
            v_n = v / np.linalg.norm(v)
            w = self.grid_space - p1  # 各格子点から始点までのベクトルを計算します
            t = np.dot(w, v_n)   # 各格子点から直線に下ろした垂線の足を求めます
            p3 = p1 + t[:, np.newaxis] * v_n
            t_min = 0  # p1に対応
            t_max = np.linalg.norm(v)  # p2に対応
            in_range = (t >= t_min) & (t <= t_max)  # 点p1とp2に対応するtの範囲内にある点だけを取り出します
            in_range_grid_space = self.grid_space[in_range]  # tの範囲内の行を取得します
            distances = np.sqrt(np.sum((in_range_grid_space - p3[in_range]) ** 2, axis=1))  # 各格子点から垂線の足までの距離を計算します
            within_threshold_grid_space = in_range_grid_space[distances <= self.threshold]  # 距離が閾値以内の行を取得します

            # w2 = within_threshold_grid_space - p1  # 各格子点から始点までのベクトルを計算します
            # t2 = np.dot(w2, v) / np.dot(v, v)  # 各格子点から直線に下ろした垂線の足を求めます
            # p3_2 = p1 + t2[:, np.newaxis] * v
            # in_range2 = (t2 >= t_min) & (t2 <= t_max)
            # distances2 = np.sqrt(np.sum((within_threshold_grid_space - p3_2[in_range2]) ** 2, axis=1))

            # within_threshold_grid_space = within_threshold_grid_space[self.threshold-2 <= distances2] 
            within_threshold_grid_space_size= within_threshold_grid_space.shape[0]  # 格子点の量の取得
            grid_space_once = np.zeros((within_threshold_grid_space_size , 8))  # 初期値の作成
            for item in range(within_threshold_grid_space_size):
                grid_space_once[item] = np.append(within_threshold_grid_space[item], theta_list[l])
            grid_table.extend(grid_space_once.tolist())
        end_time = time.time()
        print(f"処理時間: {end_time - start_time}秒")
            
        return grid_table

    def create_grid_table(self):
        for i in range(34,169):
            grid_table = self.find_gird(i) #ある角度の格子点を見つける

            with open('grid/fake_gird' + str(i) + '.csv' , 'w') as f:
                writer = csv.writer(f)
                writer.writerows(grid_table)

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
    test_c_space = calculate(-90,91,15)
    vec = test_c_space.create_C_table_once(5)
    





def main():
    c_space_maker = calculate(-90,91,15)
    c_space_maker.create_grid_table()
if __name__=="__main__":
    # main()
    test2()