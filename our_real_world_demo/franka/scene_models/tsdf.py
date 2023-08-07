from math import cos, sin
import time

import numpy as np
import open3d as o3d

class TSDFVolume(object):
    """Integration of multiple depth images using a TSDF."""

    def __init__(self, size, resolution):
        self.size = size
        self.resolution = resolution
        self.voxel_size = self.size / self.resolution
        self.sdf_trunc = 4 * self.voxel_size

        self._volume = o3d.pipelines.integration.UniformTSDFVolume(
            length=self.size,
            resolution=self.resolution,
            sdf_trunc=self.sdf_trunc,
            color_type=o3d.pipelines.integration.TSDFVolumeColorType.NoColor,
        )

    def integrate(self, depth_img, intrinsic, extrinsic, color_img=None):
        """
        Args:
            depth_img: The depth image.
            intrinsic: The intrinsic parameters of a pinhole camera model.
            extrinsics: The transform from the TSDF to camera coordinates, T_eye_task.
        """
        
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            o3d.geometry.Image(np.empty_like(depth_img)),
            o3d.geometry.Image(depth_img),
            depth_scale=1.0,
            depth_trunc=2.0,
            convert_rgb_to_intensity=False,
        )

        intrinsic = o3d.camera.PinholeCameraIntrinsic(
            width=intrinsic.width,
            height=intrinsic.height,
            fx=intrinsic.fx,
            fy=intrinsic.fy,
            cx=intrinsic.cx,
            cy=intrinsic.cy,
        )

        extrinsic = extrinsic

        self._volume.integrate(rgbd, intrinsic, extrinsic)

    def get_grid(self):
        cloud = self._volume.extract_voxel_point_cloud()
        points = np.asarray(cloud.points)
        distances = np.asarray(cloud.colors)[:, [0]]
        grid = np.zeros((1, 40, 40, 40), dtype=np.float32)
        for idx, point in enumerate(points):
            i, j, k = np.floor(point / self.voxel_size).astype(int)
            grid[0, i, j, k] = distances[idx]
        return grid

    def get_cloud(self):
        return self._volume.extract_point_cloud()
