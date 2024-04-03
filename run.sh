CUDA_VISIBLE_DEVICES=0 nohup python main.py  --lr 1e-4  --fn 32 32 --bn 32 32 --hidden 32 0  --log_dir ./log/lr_1e-4_f_32_h_32 >> lr_1e-4_f_32_h_32.log &
CUDA_VISIBLE_DEVICES=0 nohup python main.py  --lr 1e-4  --fn 32 32 --bn 32 32 --hidden 16 0  --log_dir ./log/lr_1e-4_f_32_h_16 >> lr_1e-4_f_32_h_16.log &
CUDA_VISIBLE_DEVICES=0 nohup python main.py  --lr 1e-4  --fn 32 32 --bn 32 32 --hidden 64 0  --log_dir ./log/lr_1e-4_f_32_h_64 >> lr_1e-4_f_32_h_64.log &
CUDA_VISIBLE_DEVICES=0 nohup python main.py  --lr 1e-4  --fn 32 32 --bn 32 32 --hidden 128 0 --log_dir ./log/lr_1e-4_f_32_h_128 >> lr_1e-4_f_32_h_128.log &
CUDA_VISIBLE_DEVICES=0 nohup python main.py  --lr 1e-4  --fn 16 16 --bn 16 16 --hidden 32 0 --log_dir ./log/lr_1e-4_f_16_h_32 >> lr_1e-4_f_16_h_32.log &
CUDA_VISIBLE_DEVICES=0 nohup python main.py  --lr 1e-4  --fn 16 16 --bn 16 16 --hidden 64 0 --log_dir ./log/lr_1e-4_f_16_h_64 >> lr_1e-4_f_16_h_64.log &