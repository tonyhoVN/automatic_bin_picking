# REGISTRATION MULTIPLE VIEW 
import copy
import numpy as np
import open3d as o3d
import os,sys
from scipy.spatial.transform import Rotation as R
from math import sqrt

sys.dont_write_bytecode = True
sys.path.append(os.path.abspath(os.path.dirname(__file__)))
path = os.path.abspath(os.path.join(os.path.dirname(__file__),"../data_pcl"))
sys.path.append(path)

def rot_to_zyz(matrix):
    rotation_matrix = matrix[:3,:3]
    r = R.from_matrix(np.array(rotation_matrix))
    theta3, theta2, theta1 = r.as_euler('zyz', degrees=True)
    
    return theta1, theta2, theta3


def point_to_point_color(source, target, threshold, trans_init, iteration: int = 30):
    # Estimate Normal 
    # source.estimate_normals(search_param = o3d.geometry.KDTreeSearchParamHybrid(radius = 0.002, max_nn = 10))
    # target.estimate_normals(search_param = o3d.geometry.KDTreeSearchParamHybrid(radius = 0.002, max_nn = 10))

    reg_p2p = o3d.pipelines.registration.registration_colored_icp(
        source, target, threshold, trans_init,
        o3d.pipelines.registration.TransformationEstimationForColoredICP(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration = iteration,
                                                          relative_fitness=1e-6,
                                                          relative_rmse=1e-6))

    return np.asarray(reg_p2p.transformation), reg_p2p.fitness


def point_to_point_icp(source, target, threshold, trans_init, iteration: int = 60):
    reg_p2p = o3d.pipelines.registration.registration_icp(
        source, target, threshold, trans_init,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration = iteration))

    return np.asarray(reg_p2p.transformation), reg_p2p.fitness


def point_to_plane_icp(source, target, threshold, trans_init, iteration: int = 60):
    source.estimate_normals(search_param = o3d.geometry.KDTreeSearchParamHybrid(radius = 0.005, max_nn = 10))
    target.estimate_normals(search_param = o3d.geometry.KDTreeSearchParamHybrid(radius = 0.003, max_nn = 10))
    reg_p2p = o3d.pipelines.registration.registration_icp(
        source, target, threshold, trans_init,
        o3d.pipelines.registration.TransformationEstimationPointToPlane(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration = iteration))
    return np.asarray(reg_p2p.transformation), reg_p2p.fitness

def fast_global_registration(source,target,threshold):
    source.estimate_normals(search_param = o3d.geometry.KDTreeSearchParamHybrid(radius = 0.003, max_nn = 30))
    target.estimate_normals(search_param = o3d.geometry.KDTreeSearchParamHybrid(radius = 0.003, max_nn = 30))  
    source_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
                    source,
                    o3d.geometry.KDTreeSearchParamHybrid(radius=0.005, max_nn=20))
    target_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
                    target,
                    o3d.geometry.KDTreeSearchParamHybrid(radius=0.005, max_nn=20))
    
    # result = o3d.pipelines.registration.registration_fgr_based_on_feature_matching(
    #             source, target, source_fpfh, target_fpfh,
    #             o3d.pipelines.registration.FastGlobalRegistrationOption(
    #             maximum_correspondence_distance=threshold))
    result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
    source, target, source_fpfh, target_fpfh, True, threshold,
    o3d.pipelines.registration.TransformationEstimationPointToPoint(),3, 
    [
        o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
        o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(threshold)
    ], o3d.pipelines.registration.RANSACConvergenceCriteria(100000, 0.999))
      
    return  np.asarray(result.transformation), result.fitness


def db_scan(source):
    '''
    Select clutter in the center of point cloud 
    '''
    labels = np.array(source.cluster_dbscan(eps=0.005, min_points=10))
    select_clutter = 0
    for i in range(len(source.points)):
        point = source.points[i]
        if sqrt(point[0]**2 + point[1]**2) <= 0.001:
            select_clutter = labels[i]
            break
    
    index = []
    for i in range(len(labels)):
        if labels[i] == select_clutter: index.append(i)
    clutter = source.select_by_index(index)

    return clutter

def down_size(cloud, scale:float = 1 ):
    # the scaling factor for each axis
    scale_factor = [0.001*scale, 0.001*scale, 0.001*scale] 
    
    # Create a scaling transformation
    scaling_transform = np.array([[scale_factor[0], 0, 0, 0],
                         [0, scale_factor[1], 0, 0],
                         [0, 0, scale_factor[2], 0],
                         [0, 0, 0, 1]])
    return cloud.transform(scaling_transform)

def pre_transform_mesh(cloud):
    rot_transform = np.array([[-1, 0, 0, 0],
                                  [0, 0, -1, 0],
                                  [0, -1, 0, 0],
                                  [0, 0, 0, 1]])
    cloud.transform(rot_transform)
    center = cloud.get_center()
    cloud.translate(-center)
    return cloud

def pre_process(source, init_guess, voxel_size, cutdown):
    # Voxel filter
    if voxel_size != 0:
        source = source.voxel_down_sample(voxel_size)
    
    # Translate
    source.transform(init_guess)
    
    # Filter 
    if cutdown:
        index = []
        for i in range(len(source.points)):
            if source.points[i][2] < 0: index.append(i)
        source = source.select_by_index(index)
    
    return source

def post_process(source, voxel_size, remove_outlier : bool = True):
    # Voxel filter
    if voxel_size != 0:
        source = source.voxel_down_sample(voxel_size)
    
    # Pass through filter 
    index = []
    size = 0.06
    for i in range(len(source.points)):
        if source.points[i][0] < size and source.points[i][0] > -size and source.points[i][1] < size and source.points[i][1] > -size: 
            index.append(i)
    source = source.select_by_index(index)

    # DBscan cluttering 
    # source = db_scan(source)

    # Remove outliers
    if remove_outlier:
        cl, index = source.remove_statistical_outlier(nb_neighbors=30,std_ratio=2.0)
        source = source.select_by_index(index)
    
    return source 


def global_matching(source, target):
    # REGISTER MODEL and SCENE
    best_score = 0
    transformation = np.eye(4)
    for z_angle in range(-90,90,10):
        for y_angle in range(-30,40,10):
            rot = R.from_euler('y', y_angle, degrees=True)*R.from_euler('z', z_angle, degrees=True)    
            trans_init = np.eye(4)
            trans_init[:3,:3] = rot.as_matrix()
            trans, score = point_to_plane_icp(source, target, 0.02, trans_init, 60)
            if score >= best_score:
                transformation = trans
                best_score = score  
    return transformation


def reconstruct_scene(rot_x_list, rot_y_list):
    # Load the target 
    target  = o3d.io.read_point_cloud(path + "/" + "pcl_view_0.pcd")
    target = pre_process(target, np.eye(4), 0.0015, False)

    # Stitching diff views 
    pc_rot = []

    for i in range(len(rot_x_list)):
        # Load pc
        source = o3d.io.read_point_cloud(path + "/" + "pcl_view_x" + str(i) + ".pcd")
        
        # Make stiching matrix 
        rot = R.from_euler('x', rot_x_list[i], degrees=True)
        init_guess = np.eye(4)
        init_guess[:3,:3] = rot.as_matrix()

        # Preprocess
        source = pre_process(source, init_guess, 0.0015, False)

        # Matching 
        transform_mat,_ = point_to_point_icp(source, target, 0.002, np.eye(4))

        # Add to scene
        pc_rot.append(source.transform(transform_mat))

    for i in range(len(rot_y_list)):
        # Load pc
        source = o3d.io.read_point_cloud(path + "/" + "pcl_view_y" + str(i) + ".pcd")
        
        # Make stiching matrix 
        rot = R.from_euler('y', rot_y_list[i], degrees=True)
        init_guess = np.eye(4)
        init_guess[:3,:3] = rot.as_matrix()

        # Preprocess
        source = pre_process(source, init_guess, 0.0015, False)

        # Matching 
        transform_mat,_ = point_to_point_icp(source, target, 0.002, np.eye(4))

        # Add to scene
        pc_rot.append(source.transform(transform_mat))
    
    ## Merge all pc
    scene = copy.deepcopy(target) 
    for pc in pc_rot:
        scene += pc

    # Post process 
    scene = post_process(scene, voxel_size, True)
    return scene



##### Load the sample 
voxel_size = 0.004
Sample = o3d.io.read_point_cloud(path + "/" + "component - tesselated.pcd")
Sample = pre_transform_mesh(Sample)
Sample = Sample.voxel_down_sample(0.005)
Sample.paint_uniform_color([0,1,0])


###### MAtching function
def multi_view_matching(rotate_angle_x: float, rotate_angle_y: float):

    # Load pointcloud
    target  = o3d.io.read_point_cloud(path + "/" + "pcl_view_0.pcd")
    source1 = o3d.io.read_point_cloud(path + "/" + "pcl_view_1.pcd")
    source2 = o3d.io.read_point_cloud(path + "/" + "pcl_view_2.pcd")
    threshold = 0.002 # 0.005

    # DBscan 
    # source = db_scan(source)
    # target = db_scan(target)

    # Initial guess
    rot = R.from_euler('x', rotate_angle_x, degrees=True)
    trans_init1 = np.eye(4)
    trans_init1[:3,:3] = rot.as_matrix()

    rot = R.from_euler('y', rotate_angle_y, degrees=True)
    trans_init2 = np.eye(4)
    trans_init2[:3,:3] = rot.as_matrix()

    # Preprocess
    target = pre_process(target, np.eye(4), 0.0015, False)
    source1 = pre_process(source1, trans_init1, 0.0015, False)
    source2 = pre_process(source2, trans_init2, 0.0015, False)

    # o3d.visualization.draw_geometries([target, source1, source2])

    # Merge point cloud 
    transformation1,_ = point_to_point_icp(source1, target, threshold, np.eye(4))
    transformation2,_ = point_to_point_icp(source2, target, threshold, np.eye(4))
    total = target + source2.transform(transformation2) + source1.transform(transformation1)

    # total = target + source1 + source2

    # Post process merge point cloud 
    total = post_process(total,voxel_size ,True)

    # Load sample 
    sample = copy.deepcopy(Sample)

    # Visual before matching 
    # target_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.03, origin=[0, 0, 0])
    # o3d.visualization.draw_geometries([sample, total, target_frame])

    # REGISTER MODEL and SCENE
    best_score = 0
    transformation = np.eye(4)
    z =0; y=0
    for z_angle in range(-90,90,10):
        for y_angle in range(-30,40,10):
            rot = R.from_euler('y', y_angle, degrees=True)*R.from_euler('z', z_angle, degrees=True)    
            trans_init = np.eye(4)
            trans_init[:3,:3] = rot.as_matrix()
            trans, score = point_to_plane_icp(sample, total, 0.02, trans_init, 60)
            if score >= best_score:
                transformation = trans
                best_score = score
                z = z_angle
                y = y_angle

    # FAST GLOBAL
    # transformation, best_score = fast_global_registration(sample,total,0.005)

    # Transform the sample PC
    sample.transform(transformation)

    # Visualize matching result
    # o3d.visualization.draw_geometries([total,sample])

    # Calculate ZYZ euler angle of transformation
    theta1,theta2,theta3 = rot_to_zyz(np.asarray(transformation)) # degree
    # print(np.matmul(np.matmul(R.from_euler('z', theta1, degrees=True).as_matrix(), R.from_euler('y', theta2, degrees=True).as_matrix()), R.from_euler('z', theta3, degrees=True).as_matrix()))

        # o3d.visualization.draw_geometries([total,sample])

    # Calculate XYZ translation of transformation 
    [x_trans, y_trans, z_trans] = [transformation[0][3]*1000, transformation[1][3]*1000, transformation[2][3]*1000] # milimeter
    
    return theta1, theta2, theta3, x_trans, y_trans, z_trans, transformation[:3,:3]


def single_view_matching(rotate_angle_x: float, rotate_angle_y: float):
    # Load pointcloud
    target  = o3d.io.read_point_cloud(path + "/" + "pcl_view_0.pcd")
    # total = pre_process(target, np.eye(4), voxel_size, False)
    total = post_process(target, voxel_size, False)

    # Load sample 
    sample = copy.deepcopy(Sample)

    # REGISTER MODEL and SCENE
    best_score = 0
    transformation = np.eye(4)
    z =0; y=0
    for z_angle in range(-90,90,10):
        for y_angle in range(-30,40,10):
            rot = R.from_euler('y', y_angle, degrees=True)*R.from_euler('z', z_angle, degrees=True)    
            trans_init = np.eye(4)
            trans_init[:3,:3] = rot.as_matrix()
            trans, score = point_to_plane_icp(sample, total, 0.02, trans_init,60)
            if score >= best_score:
                transformation = trans
                best_score = score
                z = z_angle
                y = y_angle

    # FAST GLOBAL
    # transformation, best_score = fast_global_registration(sample,total,0.005)

    # Transform the sample PC
    sample.transform(transformation)

    # Visualize matching result
    # o3d.visualization.draw_geometries([total,sample])

    # Calculate ZYZ euler angle of transformation
    theta1,theta2,theta3 = rot_to_zyz(np.asarray(transformation)) # degree
    # print(np.matmul(np.matmul(R.from_euler('z', theta1, degrees=True).as_matrix(), R.from_euler('y', theta2, degrees=True).as_matrix()), R.from_euler('z', theta3, degrees=True).as_matrix()))

    # Calculate XYZ translation of transformation 
    [x_trans, y_trans, z_trans] = [transformation[0][3]*1000, transformation[1][3]*1000, transformation[2][3]*1000] # milimeter
    
    return theta1, theta2, theta3, x_trans, y_trans, z_trans, transformation[:3,:3]


def combine_matching(rotate_angle_x: float, rotate_angle_y: float):

    # Load pointcloud
    target  = o3d.io.read_point_cloud(path + "/" + "pcl_view_0.pcd") # top view pc
    source1 = o3d.io.read_point_cloud(path + "/" + "pcl_view_1.pcd") # side view pc
    source2 = o3d.io.read_point_cloud(path + "/" + "pcl_view_2.pcd") # side view pc
    threshold = 0.002 # 0.005

    # DBscan 
    # source = db_scan(source)
    # target = db_scan(target)

    # Initial guess
    rot = R.from_euler('x', rotate_angle_x, degrees=True)
    trans_init1 = np.eye(4)
    trans_init1[:3,:3] = rot.as_matrix()

    rot = R.from_euler('y', rotate_angle_y, degrees=True)
    trans_init2 = np.eye(4)
    trans_init2[:3,:3] = rot.as_matrix()

    # Preprocess
    target  = pre_process(target, np.eye(4), 0.0015, False)
    source1 = pre_process(source1, trans_init1, 0.0015, False)
    source2 = pre_process(source2, trans_init2, 0.0015, False)

    # o3d.visualization.draw_geometries([target, source1, source2])

    # Merge point cloud 
    transformation1,_ = point_to_point_icp(source1, target, threshold, np.eye(4))
    transformation2,_ = point_to_point_icp(source2, target, threshold, np.eye(4))
    total = target + source2.transform(transformation2) + source1.transform(transformation1)

    # total = target + source1 + source2

    # Post process merge point cloud 
    total = post_process(total,voxel_size ,True)

    # Load sample 
    sample = copy.deepcopy(Sample)

    # Visual before matching 
    target_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.03, origin=[0, 0, 0])
    # o3d.visualization.draw_geometries([sample, total, target_frame])

    # REGISTER MODEL and SCENE with multi view 
    best_score = 0
    transformation = np.eye(4)
    z =0; y=0
    for z_angle in range(-90,90,10):
        for y_angle in range(-30,40,10):
            rot = R.from_euler('y', y_angle, degrees=True)*R.from_euler('z', z_angle, degrees=True)    
            trans_init = np.eye(4)
            trans_init[:3,:3] = rot.as_matrix()
            trans, score = point_to_plane_icp(sample, total, 0.02, trans_init, 60)
            if score >= best_score:
                transformation = trans
                best_score = score
                z = z_angle
                y = y_angle

    # Refine by single view 
    final_trans, final_score = point_to_point_icp(sample, target, 0.005, transformation, 60)

    # Transform the sample PC
    sample.transform(final_trans)

    # Visualize matching result
    # o3d.visualization.draw_geometries([total,sample])

    # Calculate ZYZ euler angle of transformation
    theta1,theta2,theta3 = rot_to_zyz(np.asarray(final_trans)) # degree

    # Calculate XYZ translation of transformation 
    [x_trans, y_trans, z_trans] = [final_trans[0][3]*1000, final_trans[1][3]*1000, final_trans[2][3]*1000] # milimeter
    
    return theta1, theta2, theta3, x_trans, y_trans, z_trans, final_trans[:3,:3] 

    
def two_view_matching(rotate_angle_x, rotate_angle_y):
    # Load pointcloud
    target  = o3d.io.read_point_cloud(path + "/" + "pcl_view_0.pcd")
    source2 = o3d.io.read_point_cloud(path + "/" + "pcl_view_2.pcd")
    threshold = 0.002 # 0.005

    # DBscan 
    # source = db_scan(source)
    # target = db_scan(target)

    # Initial guess
    rot = R.from_euler('y', rotate_angle_y, degrees=True)
    trans_init2 = np.eye(4)
    trans_init2[:3,:3] = rot.as_matrix()

    # Preprocess
    target = pre_process(target, np.eye(4), 0.0015, False)
    source2 = pre_process(source2, trans_init2, 0.0015, False)

    # o3d.visualization.draw_geometries([target, source1, source2])

    # Merge point cloud 
    transformation2,_ = point_to_point_icp(source2, target, threshold, np.eye(4))
    total = target + source2.transform(transformation2)

    # Post process merge point cloud 
    total = post_process(total,voxel_size ,True)

    # Load sample 
    sample = copy.deepcopy(Sample)

    # REGISTER MODEL and SCENE
    best_score = 0
    transformation = np.eye(4)
    z =0; y=0
    for z_angle in range(-90,90,10):
        for y_angle in range(-30,40,10):
            rot = R.from_euler('y', y_angle, degrees=True)*R.from_euler('z', z_angle, degrees=True)    
            trans_init = np.eye(4)
            trans_init[:3,:3] = rot.as_matrix()
            trans, score = point_to_plane_icp(sample, total, 0.02, trans_init, 60)
            if score >= best_score:
                transformation = trans
                best_score = score
                z = z_angle
                y = y_angle

    # Transform the sample PC
    sample.transform(transformation)

    # Visualize matching result
    # o3d.visualization.draw_geometries([total,sample])

    # Calculate ZYZ euler angle of transformation
    theta1,theta2,theta3 = rot_to_zyz(np.asarray(transformation)) # degree
    # print(np.matmul(np.matmul(R.from_euler('z', theta1, degrees=True).as_matrix(), R.from_euler('y', theta2, degrees=True).as_matrix()), R.from_euler('z', theta3, degrees=True).as_matrix()))

        # o3d.visualization.draw_geometries([total,sample])

    # Calculate XYZ translation of transformation 
    [x_trans, y_trans, z_trans] = [transformation[0][3]*1000, transformation[1][3]*1000, transformation[2][3]*1000] # milimeter
    
    return theta1, theta2, theta3, x_trans, y_trans, z_trans, transformation[:3,:3]


single_view_matching(40,-40)