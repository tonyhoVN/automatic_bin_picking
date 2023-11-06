import numpy as np
import open3d as o3d
import os,sys

sys.dont_write_bytecode = True
sys.path.append(os.path.abspath(os.path.dirname(__file__)))
path = os.path.abspath(os.path.join(os.path.dirname(__file__),"../data_pcl"))
sys.path.append(path)

def pre_transform_mesh(cloud):
    rot_transform = np.array([[-1, 0, 0, 0],
                                  [0, 0, -1, 0],
                                  [0, -1, 0, 0],
                                  [0, 0, 0, 1]])
    cloud.transform(rot_transform)
    center = cloud.get_center()
    cloud.translate(-center)
    return cloud


# Load the sample point cloud 
sample = o3d.io.read_point_cloud(path + "/" + "component - tesselated.pcd")
sample = pre_transform_mesh(sample)
sample = sample.voxel_down_sample(0.002)
sample.paint_uniform_color([0,1,0])
print(sample.get_center())


target_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.03, origin=[0, 0, 0])
o3d.visualization.draw_geometries([sample, target_frame])