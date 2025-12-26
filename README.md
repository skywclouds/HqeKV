# HqeKV
Official implementation of HqeKV: A Hybrid Mixed-Precision Quantization and Eviction Method for KV Cache Compression
### Setup
To install the required packages:
```bash
conda create -n HqeKV python=3.13
conda activate HqeKV
pip install -r requirements.txt
pip install -e .
```
Then install our CUDA implementation:
```bash
cd quant
pip install -e .
```
### inference
```bash
python pred_long_bench_hq.py --gpu_id your_id
```
You can modify the model you want to use and the proportion of each compression precision in the pred_long_bench_hq.py
The precision ratios of different models at different average compression bit-width are detailed in config/ratios.json
### Offline Precision Ratio Search
```bash
python /tests/Optuna_test.py --gpu_id your_id --avg_bit your_bit
```
