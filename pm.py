import os

from mesh.patchmatch import pm

out_dir = "./output/gen"
out_dir = os.path.join(out_dir, f"sample_{len(os.listdir(out_dir)) - 1}")
os.makedirs(out_dir, exist_ok=True)

ref_idx = 50

f = open("/home/lrz/dataset/3D_Future/split/chair.txt", "r")
texture_path_list = []
for obj_name in f.readlines():
    obj_name = obj_name.strip()
    texture_path_list.append(os.path.join("/home/lrz/dataset/3D_Future/texture", obj_name + ".png"))

pm("./output/proj/texture.png", texture_path_list[ref_idx])
