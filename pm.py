import os

from mesh.patchmatch import pm

out_dir = "./output/gen"
out_dir = os.path.join(out_dir, f"sample_{len(os.listdir(out_dir)) - 1}")
os.makedirs(out_dir, exist_ok=True)

pm("./output/proj/texture.png", "./texture/0000126.png")