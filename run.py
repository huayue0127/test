import os
os.system('bash -c "source /18942690292/huayue/anaconda3/etc/profile.d/conda.sh && conda activate && python main.py --useBN --cuda --batchSize 8 --lr 0.0005 --niter 300 --saveround 5 --save_pic_num 3 --epoch 5 --regular_epochs 5 --n_parties 6 --init_seed 646 "')

