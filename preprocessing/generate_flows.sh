for D in $(find /home/fluongo/code/usc_project/usc_data/balint/training_ready_updated/cfr_cut_mov_v2 -mindepth 2 -maxdepth 3) ; do
    echo $D 
    python convert_using_dali.py --mp4_fn $D --gpu_id 0
    python convert_using_flownet.py --mp4_fn $D --gpu_id 0
done

