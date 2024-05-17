echo $1
export CUDA_VISIBLE_DEVICES=$1
CUDA_VISIBLE_DEVICES=3 python CM-UNet/train_supervision.py -c GeoSeg/config/loveda/mambaunet.py | tee -a output_loveda.txt 
CUDA_VISIBLE_DEVICES=3 python CM-UNet/loveda_test.py -c GeoSeg/config/loveda/mambaunet.py -o fig_results/loveda/mamba --val --rgb -t 'd4' | tee -a output_loveda_test.txt

# CUDA_VISIBLE_DEVICES=3 python GeoSeg/train_supervision.py -c GeoSeg/config/loveda/abcnet.py | tee -a output_loveda_abcnet.txt 
# CUDA_VISIBLE_DEVICES=3 python GeoSeg/loveda_test.py -c GeoSeg/config/loveda/abcnet.py -o fig_results/loveda/abcnet --rgb -t 'd4' | tee -a output_loveda_test_abcnet.txt

# CUDA_VISIBLE_DEVICES=3 python GeoSeg/train_supervision.py -c GeoSeg/config/loveda/banet.py | tee -a output_loveda_banet.txt 
# CUDA_VISIBLE_DEVICES=3 python GeoSeg/loveda_test.py -c GeoSeg/config/loveda/banet.py -o fig_results/loveda/banet --rgb -t 'd4' | tee -a output_loveda_test_banet.txt

# python GeoSeg/loveda_test.py -c GeoSeg/config/loveda/banet.py -o fig_results/loveda/banet_test -t 'd4'
