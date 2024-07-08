# registration between point 
import pyvista as pv
from ruamel.yaml import YAML
import numpy as np
from copy import deepcopy
from sklearn.decomposition import PCA
from scipy.spatial.transform import Rotation as R
import open3d as o3d
import copy
import cv2
import os


class PcdCTRegistrator:
    
    def __init__(self, cfg):
        '''
        coarse to fine registration between pcd and ct:
        steps:
        1. centerize the pcd and ct
        2. pca analysis to aligh the axis
        3. coarse search with heuristic algorithm
        4. fine search with heurisitc algorithm
        
        Default CT axes:
        y up, x right, z front
        '''
        
        self.coarse_res = cfg["coarse_resolution"]
        self.fine_res = cfg["fine_resolution"]
        self.coarse_range = np.asarray(cfg["coarse_range"])
        self.fine_range = np.asarray(cfg["fine_range"])
        self.img_size = cfg["image_size"]
        self.img_org = -np.asarray(self.img_size)/2
        self.num_particles = cfg["num_particles"]
        self.top_k = cfg["top_select"]
        
        self.num_iters = cfg["num_iters"]
        
        self.pca = PCA(n_components=3)
        
        self.crop_back = cfg["crop_back"]
        
        pass
    
    
    def centerize(self, pcd):
        '''
        centerize the pcd or ct
        pcd: points cloud (pv.PolyData)
        '''
        center = np.mean(pcd.points, axis=0)
        
        result = deepcopy(pcd)
        result.points -= center
        
        return result
    
    def get_top_half(self, ct_points):
        '''
        get the top half of the ct
        ct: points cloud (pv.PolyData)
        '''
        center = np.mean(ct_points, axis=0)
        top_half = ct_points[(ct_points - center)[:, 1] > 0]
        
        return top_half
    
    
    def pca_analysis(self, data_array: np.ndarray):
        '''
        Given the data array, analyze the poses with pca, and compute coverage
        '''
        
        positions = data_array
        self.pca.fit(positions)
        axes = self.pca.components_
        eigs = self.pca.explained_variance_
        
        return axes, eigs
    
    
    def pca_adjust(self, pcd_poly):
        '''
        align the pcd and ct based on pca analysis
        '''
        pcd = pcd_poly.points # (N, 3)
        pcd_axes, pcd_eigs = self.pca_analysis(pcd)
        
        # first check the sign of the axes. var: large to small
        # largest: should toward the front direction (max distance is higher)
        proj_large = np.dot(pcd, pcd_axes[0].reshape((3, 1)))
        max_proj = np.max(proj_large)
        min_proj = np.min(proj_large)
        if np.abs(max_proj) < np.abs(min_proj):
            pcd_axes[0] = -pcd_axes[0]
            
        # smallest: should toward the up direction (front part should have proj < 0)
        proj_large = np.dot(pcd, pcd_axes[0].reshape((3, 1)))
        pcd_front = pcd[proj_large[:, 0] > 0, :]
        mean_proj_small = np.mean(np.dot(pcd_front, pcd_axes[2].reshape((3, 1))))
        if mean_proj_small > 0:
            pcd_axes[2] = -pcd_axes[2]
            
        pcd_axes[1] = np.cross(pcd_axes[2], pcd_axes[0])
        
        # largest -> z
        # smallest -> y
        # middle -> x
        result = [pcd_axes[1], pcd_axes[2], pcd_axes[0]]
        
        return result
    
    
    def pcd_align(self, pcd_poly, axes):
        '''
        rotate pcd based on the adjusted pcd axes
        '''
        axes = np.asarray(axes).T
        pcd_poly.points = (axes.T @ pcd_poly.points.T).T
        
        # crop the back parts
        z_min = np.min(pcd_poly.points[:, 2])
        points = pcd_poly.points[pcd_poly.points[:, 2] > z_min + self.crop_back]
        
        return pv.PolyData(points)
    
    
    def generate_img_from_pcd(self, pcd, res):
        '''
        generate 3d grid based on the pcd
        pcd: points cloud (pv.PolyData)
        '''
        size_array = np.asarray(self.img_size)
        num_pixels = (size_array / res).astype(int)
        img = np.zeros(num_pixels, dtype=bool)
        
        pcd_img_coords = np.floor((pcd - self.img_org) / res).astype(int)
        pcd_img_coords = np.clip(pcd_img_coords, 0, num_pixels - 1)
        img[pcd_img_coords[:, 0], pcd_img_coords[:, 1], pcd_img_coords[:, 2]] = True
        
        return img
    
    
    def quat_apply(self, quat, vec):
        '''
        apply quaternion to a vector
        quat: (4, )
        vec: (N, 3)
        '''
        q = quat
        q_1_repeat = np.repeat(q[:3].reshape((1, 3)), vec.shape[0], axis=0)
        
        q_vec = np.cross(q_1_repeat, vec) + q[3] * vec
        q_vec_0 = -vec @ q[:3].reshape((3, 1))
        
        q_vec_q_inv = - q_vec_0 * q_1_repeat + q[3] * q_vec - np.cross(q_vec, q_1_repeat)
        
        return q_vec_q_inv
    
    
    def pose_estimation_error(self, pcd, ct_img, pose, res):
        '''
        compute the error between the pcd and ct
        pcd: np array (N, 3)
        ct: np array (N, 3)
        pose: 6d vector (x, y, z, ex, ey, ez)
        res: resolution
        '''
        quat = R.from_euler('XYZ', pose[3:]).as_quat()
        # apply the pose to pcd
        pcd_rot = self.quat_apply(quat, pcd)
        pcd_trans = pcd_rot + pose[:3]
        
        # compute the error
        trans_pcd_img = self.generate_img_from_pcd(pcd_trans, res)
        higher_pcd_trans = pcd_trans + np.asarray([0, 2, 0])
        higher_trans_pcd_img = self.generate_img_from_pcd(higher_pcd_trans, res)
        err = -np.sum(np.logical_and(trans_pcd_img==1, ct_img==1))
        err += 0.7*np.sum(np.logical_and(higher_trans_pcd_img==1, ct_img==1))
        
        return err
        
        
    def pose_optimization_with_imgs(self, pcd_poly, ct_poly, p_range, res):
        '''
        optimize the pose with imgs
        pcd: poly
        ct: poly
        pose: 7d vector (x, y, z, qw, qx, qy, qz)
        '''
        mean = 0
        ps = np.random.uniform(p_range[0], p_range[1], (self.num_particles, 6))
        pcd = pcd_poly.points
        ct = ct_poly.points
        ct_img = self.generate_img_from_pcd(ct, res)
        num_ct_px = np.sum(ct_img)
        
        for iter in range(self.num_iters):
            # compute registration error
            errs = [self.pose_estimation_error(pcd, ct_img, ps[i, :], res) for i in range(self.num_particles)]
            
            # assign weight
            errs = np.asarray(errs)
            
            # update mean and variance
            mean = ps[np.argmin(errs), :]
            err_order = np.argsort(errs)
            ps_top = ps[err_order[:self.top_k], :]
            
            var = np.var(ps_top, axis=0)
            mean = mean.reshape((6,))
            std = np.sqrt(var).reshape((6,))
            
            # update particle range
            ps = np.random.uniform(mean - 2*std, mean + 2*std, (self.num_particles, 6))
            
            print(mean, std)
            
        quat = R.from_euler('XYZ', mean[3:]).as_quat()
        pcd_solved = self.quat_apply(quat, pcd) + mean[:3]
        
        return mean, pcd_solved
    
    
    def registration(self, centered_pcd, centered_ct_points):
        '''
        pcd: ply file (pv.PolyData)
        cts: list of stl files (pv.PolyData)
        '''
        pass
    
    
    def visualization(self, pcd, cts, pcd_axes=None, ct_axes=None):
        '''
        visualize the pcd and ct
        pcd: ply file (pv.PolyData)
        ct: stl file (pv.PolyData)
        '''
        p = pv.Plotter()
        p.add_mesh(pcd, color='r', opacity=0.5)
        p.add_mesh(cts, color='b', opacity=0.1)
        if pcd_axes is not None:
            for axis in pcd_axes:
                line = pv.Line((0, 0, 0), axis * 30)
                p.add_mesh(line, color='orange', line_width=4)
        if ct_axes is not None:
            for axis in ct_axes:
                line = pv.Line((0, 0, 0), axis * 30)
                p.add_mesh(line, color='green', line_width=4)
        p.show_axes()
        p.show()
        
        
def draw_registration_result(source, target, transformation):
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    source_temp.paint_uniform_color([1, 0.706, 0])
    target_temp.paint_uniform_color([0, 0.651, 0.929])
    source_temp.transform(transformation)
    o3d.visualization.draw_geometries([source_temp, target_temp],
                                      zoom=0.4459,
                                      front=[0.9288, -0.2951, -0.2242],
                                      lookat=[1.6784, 2.0612, 1.4451],
                                      up=[-0.3402, -0.9189, -0.1996])
        

def icp_registration(pcd_poly, ct_poly):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pcd_poly.points)
    ct = o3d.geometry.PointCloud()
    ct.points = o3d.utility.Vector3dVector(ct_poly.points)
    
    reg_p2p = o3d.pipelines.registration.registration_icp(
        pcd, ct, 1.0, np.eye(4),
        o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=2000))
    print(reg_p2p)
    print("Transformation is:")
    print(reg_p2p.transformation)
    
    draw_registration_result(pcd, ct, reg_p2p.transformation)
    
        


        
if __name__ == "__main__":
    
    cfg = YAML().load(open(r"pyscripts\cfg\reg_cfg.yaml", 'r'))
    registrator = PcdCTRegistrator(cfg)
    
    # read all cts
    cts = []
    for i in range(1, 6):
        ct = pv.read(r"D:\US_dataset\URS1\URS1_CT_stl\URS1_Wrapped_L"+ str(i)+" " + str(9 + 2*i) + "_001.stl")
        half_ct = registrator.get_top_half(ct.points)
        cts.append(np.asarray(half_ct))
    all_ct_p = np.concatenate(cts, axis=0)
    ct_poly = pv.PolyData(all_ct_p)
    
    # read pcd (reconstructed point cloud)
    pcd_path = r"D:\US_dataset\URS1\US_seg_recon\IDURS1-No6-R-2023-5-16-11-52\recon_remove.ply"
    pcd_poly = pv.read(pcd_path)
    
    
    
    # visualize raw data
    registrator.visualization(pcd_poly, ct_poly)
    
    # centerize the pcd and ct
    pcd_poly = registrator.centerize(pcd_poly)
    ct_poly = registrator.centerize(ct_poly)
    
    registrator.visualization(pcd_poly, ct_poly)
    
    # visualize the pca analysis of the reconstruction
    pcd_axes = registrator.pca_adjust(pcd_poly)
    registrator.visualization(pcd_poly, ct_poly, pcd_axes)
    
    # align (roughly align with the direction of the ct axes)
    pcd_poly = registrator.pcd_align(pcd_poly, pcd_axes)
    registrator.visualization(pcd_poly, ct_poly)
    
    # directly try icp
    # icp_registration(pcd_poly, ct_poly)
    
    
    # generate img
    pcd_img = registrator.generate_img_from_pcd(pcd_poly.points, registrator.fine_res)
    p = pv.Plotter()
    p.add_volume(pcd_img.astype(int))
    p.show()
    
    # rotation
    # rot_pcd = registrator.quat_apply(R.from_euler('XYZ', [0, 0, np.pi/4]).as_quat(), pcd_poly.points)
    # registrator.visualization(pv.PolyData(rot_pcd), ct_poly)
    
    # optimize
    pose, pcd_solved = registrator.pose_optimization_with_imgs(pcd_poly, ct_poly, registrator.coarse_range, registrator.coarse_res)
    # pose, pcd_solved = registrator.pose_optimization_with_imgs(pcd_poly, ct_poly, registrator.fine_range, registrator.fine_res)
    
    pcd_solved_poly = pv.PolyData(pcd_solved)
    registrator.visualization(pcd_solved_poly, ct_poly)
    
    # icp
    icp_registration(pcd_solved_poly, ct_poly)
    
    
    
   
    