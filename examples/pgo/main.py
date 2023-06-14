import torch
from lietorch import SO3, SE3, LieGroupParameter

import argparse
import numpy as np
import time
import torch.optim as optim
import torch.nn.functional as F


def draw(verticies):
    """ draw pose graph """
    import open3d as o3d

    n = len(verticies)
    points = np.array([x[1][:3] for x in verticies])
    lines = np.stack([np.arange(0,n-1), np.arange(1,n)], 1)

    line_set = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(points),
        lines=o3d.utility.Vector2iVector(lines),
    )
    o3d.visualization.draw_geometries([line_set])

def info2mat(info):
    mat = np.zeros((6,6))
    ix = 0
    for i in range(mat.shape[0]):
        mat[i,i:] = info[ix:ix+(6-i)]
        mat[i:,i] = info[ix:ix+(6-i)]
        ix += (6-i)

    return mat

def read_g2o(fn):
    verticies, edges = [], []
    # 使用with语句打开文件后，在语句块结束时会自动关闭文件，无需手动调用close()方法。
    with open(fn) as f:
        for line in f:
            line = line.split()
            if line[0] == 'VERTEX_SE3:QUAT':
                v = int(line[1])        #位姿ID
                pose = np.array(line[2:], dtype=np.float32)
                verticies.append([v, pose]) # 添加顶点

            elif line[0] == 'EDGE_SE3:QUAT':
                u = int(line[1])    #位姿1 ID
                v = int(line[2])    #位姿2 ID
                pose = np.array(line[3:10], dtype=np.float32)   # 位姿
                info = np.array(line[10:], dtype=np.float32)    # 信息矩阵

                info = info2mat(info)   # 转换成矩阵
                edges.append([u, v, pose, info, line])  # 添加边

    return verticies, edges

def write_g2o(pose_graph, fn):
    import csv
    verticies, edges = pose_graph
    with open(fn, 'w') as f:
        writer = csv.writer(f, delimiter=' ')
        for (v, pose) in verticies:
            row = ['VERTEX_SE3:QUAT', v] + pose.tolist()
            writer.writerow(row)
        for edge in edges:
            writer.writerow(edge[-1])

def reshaping_fn(dE, b=1.5):
    """ Reshaping function from "Intrinsic consensus on SO(3), Tron et al."""
    ang = dE.log().norm(dim=-1)
    err = 1/b - (1/b + ang) * torch.exp(-b*ang)
    return err.sum()

def gradient_initializer(pose_graph, n_steps=500, lr_init=0.2):
    """ Riemannian Gradient Descent 黎曼梯度下降"""

    verticies, edges = pose_graph

    # edge indicies (ii, jj)
    ii = np.array([x[0] for x in edges])    # 列表推导式，获得边的第一个顶点的id
    jj = np.array([x[1] for x in edges])    # 列表推导式，获得边的第二个顶点的id
    ii = torch.from_numpy(ii).cuda()        # 将numpy的array转为torch的tensor类型，并将其移动到Gpu上去
    jj = torch.from_numpy(jj).cuda()

    Eij = np.stack([x[2][3:] for x in edges])   # 将所有边的2顶点的位姿(平移+四元数)合并为1个数组
    Eij = SO3(torch.from_numpy(Eij).float().cuda()) # 先转为tensor，再创建SO(3)类

    R = np.stack([x[1][3:] for x in verticies]) # x[0]id,x[1]7个值取后面4个四元数构建旋转矩阵
    R = SO3(torch.from_numpy(R).float().cuda())
    R = LieGroupParameter(R)

    # use gradient descent with momentum
    optimizer = optim.SGD([R], lr=lr_init, momentum=0.5)    # 定义torch的随机梯度下降sgd优化器

    start = time.time()
    for i in range(n_steps):
        optimizer.zero_grad()   # 将模型参数的梯度置零，以准备接收新的梯度。

        for param_group in optimizer.param_groups:
            param_group['lr'] = lr_init * .995**i

        # rotation error
        dE = (R[ii].inv() * R[jj]) * Eij.inv()
        loss = reshaping_fn(dE)

        loss.backward()     # 根据损失值计算参数的梯度
        optimizer.step()    # 根据梯度更新模型的参数

        if i%25 == 0:
            print(i, lr_init * .995**i, loss.item())

    # convert rotations to pose3
    quats = R.group.data.detach().cpu().numpy()

    for i in range(len(verticies)):
        verticies[i][1][3:] = quats[i]

    return verticies, edges


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--problem', help="input pose graph optimization file (.g2o format)")
    args = parser.parse_args()

    output_path = args.problem.replace('.g2o', '_rotavg.g2o')   # problem是上面的参数名，replace()函数是argparse库的
    input_pose_graph = read_g2o(args.problem)                   # args.problem：保存的是.g2o的地址。返回所有的顶点和边

    rot_pose_graph = gradient_initializer(input_pose_graph)     # 返回优化好后的顶点和边
    write_g2o(rot_pose_graph, output_path)

