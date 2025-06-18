cls_name=$1
ref_idx=$2
tar_idx=$3

base_data_path="./dataset"
data_path="${base_data_path}/${cls_name}"

ref_mesh="real_data/0/mesh.obj" 
ref_texture="real_data/0/texture.png"
tar_mesh="${data_path}/${tar_idx}/mesh.obj"

if [ ! -d ".cache" ]; then
    mkdir .cache
fi

starttime=`date +'%Y-%m-%d %H:%M:%S'`

python uv_gen.py ref_mesh=$ref_mesh ref_texture=$ref_texture tar_mesh=$tar_mesh
python uv_project.py tar_mesh=$tar_mesh
python uv_gen_up.py strength=0.4 ref_mesh=$ref_mesh ref_texture=$ref_texture tar_mesh=$tar_mesh tar_texture=output/texture_l.png
python uv_project_up.py tar_mesh=$tar_mesh
python refine.py strength=0.3 ref_mesh=$ref_mesh ref_texture=$ref_texture tar_mesh=$tar_mesh tar_texture=output/texture_h.png
 
endtime=`date +'%Y-%m-%d %H:%M:%S'`
start_seconds=$(date --date="$starttime" +%s);
end_seconds=$(date --date="$endtime" +%s);
echo "run time "$((end_seconds-start_seconds))"s"

echo "finish!"
if [ ! -d "results" ]; then
    mkdir results
fi
rename="results/output-${cls_name}-${ref_idx}-${tar_idx}"
mkdir -p $rename
mv output $rename
mv .cache $rename