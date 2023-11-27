# REGISTRATION MULTIPLE VIEW 
import copy
import numpy as np
import open3d as o3d
import os,sys, time
from scipy.spatial.transform import Rotation as R
from math import sqrt

# Setup path
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

    return np.asarray(reg_p2p.transformation), reg_p2p.inlier_rmse


def point_to_point_icp(source, target, threshold, trans_init, iteration: int = 60):
    # loss = o3d.pipelines.registration.TukeyLoss(k=1.0)
    reg_p2p = o3d.pipelines.registration.registration_icp(
        source, target, threshold, trans_init,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration = iteration))

    return np.asarray(reg_p2p.transformation), reg_p2p.inlier_rmse, reg_p2p.fitness


def point_to_plane_icp(source, target, threshold, trans_init, iteration: int = 60):
    loss = o3d.pipelines.registration.TukeyLoss(k=1.0)
    source.estimate_normals(search_param = o3d.geometry.KDTreeSearchParamHybrid(radius = 0.003, max_nn = 5))
    target.estimate_normals(search_param = o3d.geometry.KDTreeSearchParamHybrid(radius = 0.003, max_nn = 5))
    reg_p2p = o3d.pipelines.registration.registration_icp(
        source, target, threshold, trans_init,
        o3d.pipelines.registration.TransformationEstimationPointToPlane(loss),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration = iteration))
    return np.asarray(reg_p2p.transformation), reg_p2p.inlier_rmse, reg_p2p.fitness


def fast_global_registration(source,target):
    source.estimate_normals(search_param = o3d.geometry.KDTreeSearchParamHybrid(radius = voxel_size*3, max_nn = 30))
    target.estimate_normals(search_param = o3d.geometry.KDTreeSearchParamHybrid(radius = 0.003*3, max_nn = 30))  
    
    ## Find FPFH
    source_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
                    source,
                    o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size*5, max_nn=100))
    target_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
                    target,
                    o3d.geometry.KDTreeSearchParamHybrid(radius=0.003*5, max_nn=100))
    
    # Fast global localization 
    result = o3d.pipelines.registration.registration_fast_based_on_feature_matching(
                source, target, source_fpfh, target_fpfh,
                o3d.pipelines.registration.FastGlobalRegistrationOption(
                maximum_correspondence_distance=0.01,
                iteration_number = 60))

    # ## Ransac
    # result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
    # source, target, source_fpfh, target_fpfh, True, 0.04,
    # o3d.pipelines.registration.TransformationEstimationPointToPoint(),3, 
    # [
    #     o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.1),
    #     o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(0.04)
    # ], o3d.pipelines.registration.RANSACConvergenceCriteria(100000, 0.999))
      
    return  np.asarray(result.transformation), result.fitness


def db_scan(source):
    '''
    Select clutter in the center of point cloud 
    '''
    labels = np.array(source.cluster_dbscan(eps=0.001, min_points=5))
    select_clutter = 1
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
    rot_transform = np.array([[0.001, 0, 0, 0],
                              [0, 0.001, 0, 0],
                              [0, 0, 0.001, 0],
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
    size = 0.03
    for i in range(len(source.points)):
        if source.points[i][0] < size \
            and source.points[i][0] > -size \
            and source.points[i][1] < size \
            and source.points[i][1] > -size \
            and source.points[i][2] < 0.025: 
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
    best_score = 999
    best_fitness = 0
    transformation = np.eye(4)
    for z_angle in range(-90,90,10):
        for y_angle in range(-0,10,10):
            rot = R.from_euler('y', y_angle, degrees=True)*R.from_euler('z', z_angle, degrees=True)    
            trans_init = np.eye(4)
            trans_init[:3,:3] = rot.as_matrix()
            trans, score, fitness = point_to_plane_icp(source, target, 0.01, trans_init, 60)
            if score <= best_score:
                transformation = trans
                best_score = score 
                best_fitness = fitness 
    return transformation, best_score



def reconstruct_scene(rot_x_list:list = [], rot_y_list:list = []):
    # Load the target 
    target  = o3d.io.read_point_cloud(path + "/" + "pcl_view_0.pcd")
    target = pre_process(target, np.eye(4), 0.002, False)

    # Stitching diff views top construct scene 
    scene = copy.deepcopy(target)

    for i in range(len(rot_x_list)):
        # Load pc
        source = o3d.io.read_point_cloud(path + "/" + "pcl_view_x_" + str(i) + ".pcd")
        
        # Make stiching matrix 
        rot = R.from_euler('x', rot_x_list[i], degrees=True)
        init_guess = np.eye(4)
        init_guess[:3,:3] = rot.as_matrix()

        # Preprocess
        source = pre_process(source, init_guess, 0.002, False)

        # Matching 
        transform_mat,_,_ = point_to_point_icp(source, target, 0.003, np.eye(4))

        # Add to scene
        scene += source.transform(transform_mat)

    for i in range(len(rot_y_list)):
        # Load pc
        source = o3d.io.read_point_cloud(path + "/" + "pcl_view_y_" + str(i) + ".pcd")
        
        # Make stiching matrix 
        rot = R.from_euler('y', rot_y_list[i], degrees=True)
        init_guess = np.eye(4)
        init_guess[:3,:3] = rot.as_matrix()

        # Preprocess
        source = pre_process(source, init_guess, 0.002, False)

        # Matching 
        transform_mat,_,_ = point_to_point_icp(source, target, 0.003, np.eye(4))

        # Add to scene
        scene += source.transform(transform_mat)
        
    
    # Post process 
    scene = post_process(scene, voxel_size, True)
    return scene



##### Load the sample 
voxel_size = 0.002 # 0.004
Sample = o3d.io.read_point_cloud(path + "/" + "object.pcd")
Sample = pre_transform_mesh(Sample)
Sample = Sample.voxel_down_sample(0.002) #0.005
Sample.paint_uniform_color([0,0,0.5])
###### Matching function
def multi_view_matching(rotate_angle_x: list, rotate_angle_y: list):
    # Reconstruct scene 
    scene = reconstruct_scene(rotate_angle_x, rotate_angle_y)

    # Load point cloud of sample 
    sample = copy.deepcopy(Sample)
    # o3d.visualization.draw_geometries([scene,sample])

    # Global matching
    start_time = time.time()

    transformation,_ = global_matching(source = sample, target = scene)

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"mathcing time: {elapsed_time} ")

    # Transform the sample PC
    sample.transform(transformation)

    # Visualize matching result
    o3d.visualization.draw_geometries([scene,sample])

    # Calculate ZYZ euler angle of transformation
    theta1,theta2,theta3 = rot_to_zyz(np.asarray(transformation)) # degree

    # Calculate XYZ translation of transformation 
    [x_trans, y_trans, z_trans] = [transformation[0][3]*1000, transformation[1][3]*1000, transformation[2][3]*1000] # milimeter
    
    return theta1, theta2, theta3, x_trans, y_trans, z_trans, transformation[:3,:3]


def two_view_matching(rotate_angle_x: list, rotate_angle_y: list):
    # Reconstruct scene 
    scene = reconstruct_scene([],[rotate_angle_y[0]])

    # Load point cloud of sample 
    sample = copy.deepcopy(Sample)

    # Global matching
    transformation,_ = global_matching(source = sample, target = scene)
    # transformation = fast_global_registration(source = sample, target = scene)

    # Transform the sample PC
    sample.transform(transformation)

    # Visualize matching result
    # o3d.visualization.draw_geometries([scene,sample])

    # Calculate ZYZ euler angle of transformation
    theta1,theta2,theta3 = rot_to_zyz(np.asarray(transformation)) # degree

    # Calculate XYZ translation of transformation 
    [x_trans, y_trans, z_trans] = [transformation[0][3]*1000, transformation[1][3]*1000, transformation[2][3]*1000] # milimeter
    
    return theta1, theta2, theta3, x_trans, y_trans, z_trans, transformation[:3,:3]


def single_view_matching(rotate_angle_x: list, rotate_angle_y: list):
    # Reconstruct scene 
    scene = reconstruct_scene()

    # Load point cloud of sample 
    sample = copy.deepcopy(Sample)

    # Global matching
    transformation,_ = global_matching(source = sample, target = scene)
    # transformation,_ = fast_global_registration(source = sample, target = scene)
    # best_score = 0
    # transformation = np.eye(4)
    # for z_angle in range(-90,90,10):
    #     for y_angle in range(-30,30,10):
    #         rot = R.from_euler('y', y_angle, degrees=True)*R.from_euler('z', z_angle, degrees=True)    
    #         trans_init = np.eye(4)
    #         trans_init[:3,:3] = rot.as_matrix()
    #         trans, score = point_to_point_icp(sample, scene, 0.04, trans_init, 60)
    #         if score >= best_score:
    #             transformation = trans
    #             best_score = score  

    # Transform the sample PC
    sample.transform(transformation)

    # Visualize matching result
    o3d.visualization.draw_geometries([scene,sample])

    # Calculate ZYZ euler angle of transformation
    theta1,theta2,theta3 = rot_to_zyz(np.asarray(transformation)) # degree

    # Calculate XYZ translation of transformation 
    [x_trans, y_trans, z_trans] = [transformation[0][3]*1000, transformation[1][3]*1000, transformation[2][3]*1000] # milimeter
    
    return theta1, theta2, theta3, x_trans, y_trans, z_trans, transformation[:3,:3]


def combine_matching(rotate_angle_x: list, rotate_angle_y: list):
    # Load the top view
    top  = o3d.io.read_point_cloud(path + "/" + "pcl_view_0.pcd")
    top = post_process(top, voxel_size, False)
    top = db_scan(top)
    
    # Reconstruct scene 
    scene = reconstruct_scene([rotate_angle_x[0]],[rotate_angle_y[0]])

    # Load point cloud of sample 
    sample = copy.deepcopy(Sample)

    # Global matching
    transformation,_ = global_matching(source = sample, target = scene)
    # transformation,_ = fast_global_registration(source = sample, target = scene)
    final_trans,rmsq,fitness = transformation,999,0


     # Refine by single view 
    final_trans,rmsq,fitness = point_to_point_icp(sample, top, 0.005, final_trans, 60)

    # print(final_trans)

    # Transform the sample PC
    sample.transform(final_trans)

    # Visualize matching result
    # o3d.visualization.draw_geometries([scene,sample])

    # Calculate ZYZ euler angle of transformation
    theta1,theta2,theta3 = rot_to_zyz(np.asarray(final_trans)) # degree
    final_rot = R.from_euler('zyz',[theta1,0,theta3],degrees=True).as_matrix()

    # Calculate XYZ translation of transformation 
    [x_trans, y_trans, z_trans] = [final_trans[0][3]*1000, final_trans[1][3]*1000, final_trans[2][3]*1000] # milimeter
    

    return theta1, theta2, theta3, x_trans, y_trans, z_trans, final_rot 

# combine_matching([40],[-40])