# HqeKV

Official implementation of HqeKV: Towards Hybrid Quantization and Eviction for KV Cache in Long-Context LLM Inference

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

### Inference

```bash
python pred_long_bench_hq.py --gpu_id your_gid
```

You can modify the model you want to use and the proportion of each compression precision in the pred_long_bench_hq.py
The precision ratios of different models at different average compression bit-width are detailed in config/ratios.json

### Offline Precision Ratio Search

```bash
python /tests/Optuna_test.py --gpu_id your_gid --avg_bit your_bit
```

### Memory Usage and Throughput

```bash
python /tests/batch_size_test.py --batch_size your_bsz --gpu_id your_gid --strategies_id your_sid
```

You can modify the /tests/batch_size_test.py to change the input_length.
