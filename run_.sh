cls_name=$1
ref_idx=$2
tar_idx=$3

base_data_path="./dataset"
data_path="${base_data_path}/${cls_name}"

ref_mesh="${data_path}/${ref_idx}/mesh.obj" 
ref_texture="${data_path}/${ref_idx}/texture.png" 
tar_mesh="${data_path}/${tar_idx}/mesh.obj"

if [ ! -d ".cache" ]; then
    mkdir .cache
fi
# python uv_gen.py ref_mesh=$ref_mesh ref_texture=$ref_texture tar_mesh=$tar_mesh
# python uv_project.py tar_mesh=$tar_mesh
# python uv_gen_up.py strength=0.4 ref_mesh=$ref_mesh ref_texture=$ref_texture tar_mesh=$tar_mesh tar_texture=output/texture_l.png
# python uv_project_up.py tar_mesh=$tar_mesh
python refine.py strength=0.4 ref_mesh=$ref_mesh ref_texture=$ref_texture tar_mesh=$tar_mesh tar_texture=output/texture_h.png

echo "finish!"
if [ ! -d "results" ]; then
    mkdir results
fi
rename="results/output-${cls_name}-${ref_idx}-${tar_idx}"
mkdir -p $rename
mv output $rename
mv .cache $rename