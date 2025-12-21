## 🚀 Inference Examples

Rectified SpaAttn currently supports Wan2.2 (TI2V / I2V / T2V), CogVideoX1.5 (I2V / T2V), HunyuanVideo, Wan2.1 (I2V / T2V), and Flux.1-dev. You can use Rectified SpaAttn as follows:

#### Inference Parameters 

- --sa_drop_rate: Sparsity ratio, specifies the ratio of attention blocks removed during sparse attention. Higher values correspond to more aggressive sparsification.
- --enable_teacache: Caching flag, activate model–cache fusion, yielding faster inference.
- --use_sage: SageAttention flag, activate SageAttention1 fusion, yielding faster inference.

### HunyuanVideo (720×1280, 128 frames)

The running scripts are:
```bash
python scripts/main_hunyuan.py --sa_drop_rate 0.8 # 2.50× speedup
python scripts/main_hunyuan.py --sa_drop_rate 0.8 --enable_teacache # 5.24× speedup
```

### Wan 2.1 (720×1280, 81 frames)
The running scripts are:
```bash
# Text-to-Video
python scripts/main_wan21t2v.py --sa_drop_rate 0.75 # 1.68× speedup
python scripts/main_wan21t2v.py --sa_drop_rate 0.75 --enable_teacache # 4.61× speedup

# Image-to-Video
python scripts/main_wan21i2v.py --sa_drop_rate 0.75 # 1.81× speedup
python scripts/main_wan21i2v.py --sa_drop_rate 0.75 --enable_teacache # 8.97× speedup
```

### Flux.1-dev (4096×4096)

The running scripts are:
```bash
python scripts/main_upflux.py --sa_drop_rate 0.9 # 1.60× speedup
python scripts/main_upflux.py --sa_drop_rate 0.9 --enable_teacache # 4.15× speedup
```

### CogVideoX1.5 (768×1280, 81 frames)
The running scripts are:
```bash
# Text-to-Video
python scripts/main_cogvideox.py --generate_type t2v --sa_drop_rate 0.85 # 1.76× speedup
python scripts/main_cogvideox.py --generate_type t2v --sa_drop_rate 0.85 --enable_teacache # 2.97× speedup

# Image-to-Video
python scripts/main_cogvideox.py --generate_type i2v --sa_drop_rate 0.75 # 1.60× speedup
python scripts/main_cogvideox.py --generate_type i2v --sa_drop_rate 0.75 --enable_teacache # 2.90× speedup
```

### Wan 2.2
The running scripts are:
```bash
# TI2V-5B (704×1280, 121 frames)
python scripts/main_wan22ti2v.py --sa_drop_rate 0.75 # 1.28× speedup
python scripts/main_wan22ti2v.py --sa_drop_rate 0.75 --enable_teacache # 1.83× speedup

# T2V-A14B (720×1280, 81 frames)
python scripts/main_wan22t2v.py --sa_drop_rate 0.85 # 1.87× speedup
python scripts/main_wan22t2v.py --sa_drop_rate 0.75 --enable_teacache # 3.50× speedup

# I2V-A14B (720×1280, 81 frames)
python scripts/main_wan22i2v.py --sa_drop_rate 0.85 # 2.08× speedup
python scripts/main_wan22i2v.py --sa_drop_rate 0.75 --enable_teacache # 5.36× speedup
```
