# ref_idx=1a0bc9ab92c915167ae33d942430658c
# tar_idx=1a0c91c02ef35fbe68f60a737d94994a
ref_idx=$1
tar_idx=$2

tar_mesh="resource/dataset/shapenetv1/02958343/${tar_idx}/model.obj"
# tar_mesh="/home/lrz/baseline/Paint3D/eval/tar.obj"

if [ ! -d ".cache" ]; then
    mkdir .cache
fi

starttime=`date +'%Y-%m-%d %H:%M:%S'`

python uv_gen.py ref_idx=$ref_idx tar_mesh=$tar_mesh tar_idx=$tar_idx
python uv_project.py tar_mesh=$tar_mesh
python uv_gen_up.py strength=0.4 ref_idx=$ref_idx tar_mesh=$tar_mesh tar_texture=output/texture_l.png
python uv_project_up.py tar_mesh=$tar_mesh
python refine.py strength=0.3 ref_idx=$ref_idx tar_mesh=$tar_mesh tar_texture=output/texture_h.png
 
endtime=`date +'%Y-%m-%d %H:%M:%S'`
start_seconds=$(date --date="$starttime" +%s);
end_seconds=$(date --date="$endtime" +%s);
echo "run time "$((end_seconds-start_seconds))"s"

echo "finish!"
if [ ! -d "results" ]; then
    mkdir results
fi
rename="results/output-car-${ref_idx}-${tar_idx}"
mkdir -p $rename
mv output $rename
mv .cache $rename