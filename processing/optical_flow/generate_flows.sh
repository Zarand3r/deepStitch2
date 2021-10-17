# for D in $(find /mnt/md1/richard_bao/balint_data/classification_data -mindepth 2 -maxdepth 3) ; do
#     echo $D 
#     # python convert_using_dali.py --mp4_fn $D --gpu_id 0
#     python convert_using_flownet.py --mp4_fn $D
# done

for D in $(find /mnt/md1/richard_bao/tubes_data2/classification_data -type f -name '*[[:space:]]*' -mindepth 2 -maxdepth 2) ; do
    # echo $D 
    python convert_using_flownet.py --mp4_fn $D --gpu_id 0
done
