for D in $(find /central/groups/tensorlab/rbao/usc_data/classification_data -mindepth 2 -maxdepth 3) ; do
    echo $D 
    # python convert_using_dali.py --mp4_fn $D --gpu_id 0
    python convert_using_flownet.py --mp4_fn $D --gpu_id 0
done

