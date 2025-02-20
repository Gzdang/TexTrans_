# from https://github.com/threedle/text2mesh
import kaolin as kal
import torch
import numpy as np
from torchvision.transforms import Resize


class Render(torch.nn.Module):
    def __init__(self, render_size=(512, 512), image_size=(512, 512), interpolation_mode="nearest", device="cuda"):
        assert interpolation_mode in [
            "nearest",
            "bilinear",
            "bicubic",
        ], f"no interpolation mode {interpolation_mode}"
        super().__init__()

        #  Generate perspective projection matrix for a given camera fovy angle.
        self.device = device
        self.camera_projection = kal.render.camera.generate_perspective_projection(np.pi / 4).to(device)

        self.interpolation_mode = interpolation_mode
        self.render_size = render_size
        self.resizer = Resize(image_size, antialias=True)

    def get_camera_from_view(self, elev, azim, r=3.0, look_at_height=0.0):
        x = r * np.sin(elev) * np.sin(azim)
        y = r * np.cos(elev)
        z = r * np.sin(elev) * np.cos(azim)

        if isinstance(elev, torch.Tensor) and isinstance(azim, torch.Tensor):
            pos = torch.stack([x, y, z], 1)
        else:
            pos = torch.tensor([x, y, z], dtype=torch.float32).unsqueeze(0)

        look_at = torch.zeros_like(pos)
        look_at[:, 1] = look_at_height
        direction = torch.tensor([0.0, 1.0, 0.0], dtype=torch.float32).unsqueeze(0).repeat(pos.shape[0], 1)
        # if elev == 0 and azim == 0:
        #     direction = torch.tensor([0.0, 0.0, 1.0], dtype=torch.float32).unsqueeze(0)

        camera_proj = kal.render.camera.generate_transformation_matrix(pos, look_at, direction)
        return camera_proj

    def normalize_depth(self, depth_map):
        assert depth_map.max() <= 0.0, "depth map should be negative"
        object_mask = depth_map != 0
        # depth_map[object_mask] = (depth_map[object_mask] - depth_map[object_mask].min()) / (
        #             depth_map[object_mask].max() - depth_map[object_mask].min())
        # depth_map = depth_map ** 4
        min_val = 0.5
        depth_map[object_mask] = (
            (1 - min_val)
            * (depth_map[object_mask] - depth_map[object_mask].min())
            / (depth_map[object_mask].max() - depth_map[object_mask].min())
        ) + min_val
        # depth_map = (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min())
        # depth_map[depth_map == 1] = 0 # Background gets largest value, set to 0

        return depth_map

    def forward(self, mesh, texture, camera):
        vertex_matrix = mesh.vertices
        face_matrix = mesh.faces
        face_uv_matrix = mesh.face_uv_matrix
        face_normal_matrix = mesh.face_normal_matrix

        (
            face_vertices_camera,
            face_vertices_image,
            face_normals,
        ) = kal.render.mesh.prepare_vertices(
            vertex_matrix,
            face_matrix,
            self.camera_projection,
            camera_transform=camera,
        )

        view_uv, face_idx_uv = kal.render.mesh.rasterize(
            *self.render_size[::-1],
            face_vertices_camera[:, :, :, -1],
            face_vertices_image,
            face_uv_matrix,
        )
        view_normal, _ = kal.render.mesh.rasterize(
            self.render_size[1],
            self.render_size[0],
            face_vertices_camera[:, :, :, -1],
            face_vertices_image,
            face_normal_matrix,
        )
        view_depth, _ = kal.render.mesh.rasterize(
            self.render_size[1],
            self.render_size[0],
            face_vertices_camera[:, :, :, -1],
            face_vertices_image,
            face_vertices_camera[:, :, :, -1:],
        )
        view_depth = self.normalize_depth(view_depth)
        view_mask = (face_idx_uv > -1).float()[..., None]
        view_image = kal.render.mesh.texture_mapping(view_uv, texture, mode=self.interpolation_mode)

        # view_image = view_image * view_mask
        view_image = view_image * view_mask + (1-view_mask)

        view_uv = view_uv * view_mask
        view_depth = view_depth * view_mask
        view_normal = (view_normal + 1.0) * 0.5 * view_mask

        view_image = view_image.permute(0, 3, 1, 2)
        view_mask = view_mask.permute(0, 3, 1, 2)
        view_depth = view_depth.permute(0, 3, 1, 2)
        view_normal = view_normal.permute(0, 3, 1, 2)

        return {
            "image": self.resizer(view_image),
            "depth": self.resizer(view_depth),
            "mask": self.resizer(view_mask),
            "normal": self.resizer(view_normal),
        }

    def render_multi_view_texture(
        self,
        verts,
        faces,
        uv_face_attr,
        texture_map,
        elev=0,
        azim=0,
        radius=2,
        look_at_height=0.0,
        dims=None,
        background_type="none",
        render_cache=None,
    ):
        dims = self.render_size if dims is None else dims

        batch_size = len(elev)
        uv_face_attr = uv_face_attr.repeat(batch_size, 1, 1, 1)
        texture_map = texture_map.repeat(batch_size, 1, 1, 1)

        if render_cache is None:

            camera_transform = self.get_camera_from_view(
                torch.tensor(elev), torch.tensor(azim), r=radius, look_at_height=look_at_height
            ).to(self.device)
            face_vertices_camera, face_vertices_image, face_normals = kal.render.mesh.prepare_vertices(
                verts.to(self.device), faces.to(self.device), self.camera_projection, camera_transform=camera_transform
            )

            uv_features, face_idx = kal.render.mesh.rasterize(
                dims[1], dims[0], face_vertices_camera[:, :, :, -1], face_vertices_image, uv_face_attr
            )
            uv_features = uv_features.detach()

            depth, _ = kal.render.mesh.rasterize(
                dims[1],
                dims[0],
                face_vertices_camera[:, :, :, -1],
                face_vertices_image,
                face_vertices_camera[:, :, :, -1:],
            )

            depth = self.normalize_depth(depth)

            mask = (face_idx > -1).float()[..., None]
            normal = face_normals[0][face_idx, :]

            depth = depth.permute(0, 3, 1, 2)
            normal = normal.permute(0, 3, 1, 2)
            mask = mask.permute(0, 3, 1, 2)

        else:
            # logger.info('Using render cache')
            uv_features, depth, mask, normal = (
                render_cache["uv_features"],
                render_cache["depth"],
                render_cache["mask"],
                render_cache["normal"],
            )

        image = kal.render.mesh.texture_mapping(uv_features, texture_map, mode=self.interpolation_mode)
        image = image.permute(0, 3, 1, 2)
        image = image * mask

        render_cache = {"uv_features": uv_features, "depth": depth, "mask": mask, "normal": normal}

        return image, mask, depth, normal, render_cache
