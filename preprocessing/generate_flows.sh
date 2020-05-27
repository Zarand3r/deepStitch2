for D in $(find /home/fluongo/code/usc_project/usc_data/balint/training_ready/cutting_model_allExamps -mindepth 2 -maxdepth 3) ; do
    echo $D 
    python convert_using_dali.py --mp4_fn $D --gpu_id 1
    python convert_using_flownet.py --mp4_fn $D --gpu_id 1

done