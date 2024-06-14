import torch
import kaolin as kal

from mesh.utils import import_mesh


class Mesh(torch.nn.Module):
    def __init__(self, obj_path, device="cuda"):
        super().__init__()

        mesh = import_mesh(obj_path, with_normals=True, with_materials=True)

        self.vertices = mesh.vertices.to(device)
        self.faces = mesh.faces.to(device)
        self.vertex_normals = mesh.normals.to(device)
        self.face_normals = mesh.face_normals_idx.to(device)
        self.vertex_uvs = mesh.uvs.to(device)
        self.face_uvs = mesh.face_uvs_idx.to(device)

        self.normalize_mesh(scale_rate=1.2)

        self.face_xyz_matrix = kal.ops.mesh.index_vertices_by_faces(
            self.vertices.unsqueeze(0), self.faces.long()
        ).to(device)
        self.face_normal_matrix = kal.ops.mesh.index_vertices_by_faces(
            self.vertex_normals.unsqueeze(0), self.face_normals.long()
        )
        self.face_uv_matrix = kal.ops.mesh.index_vertices_by_faces(self.vertex_uvs.unsqueeze(0), self.face_uvs.long()).to(device)

    def to(self, device):
        self.vertices = self.vertices.to(device)
        self.faces = self.faces.to(device)
        self.vertex_normals = self.vertex_normals.to(device)
        self.face_normals = self.face_normals.to(device)
        self.vertex_uvs = self.vertex_uvs.to(device)
        self.face_uvs = self.face_uvs.to(device)
        self.face_xyz_matrix = self.face_xyz_matrix.to(device)
        # self.face_normal_matrix = self.face_normal_matrix.to(device)
        self.face_uv_matrix = self.face_uv_matrix.to(device)
        return self

    def normalize_mesh(self, scale_rate=1, dy=0):
        vertices = self.vertices
        
        center = (torch.min(vertices, dim=0)[0] + torch.max(vertices, dim=0)[0])/2
        vertices = vertices - center
        scale = torch.max(torch.norm(vertices, p=2, dim=1))
        vertices = vertices / scale

        vertices *= scale_rate
        vertices[:, 1] += dy
        self.vertices = vertices
