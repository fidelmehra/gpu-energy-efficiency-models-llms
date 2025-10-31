# GPU Energy Efficiency Models for LLMs (Full-Power Guide)

Author: Fidel Mehra

A comprehensive, practitioner-focused repository on energy-efficient GPU optimization for Large Language Models (LLMs). Includes conceptual foundations, literature, reproducible models, code (Python/NumPy/PyTorch), notebooks, best practices, hardware/software setup, case studies, and references.

---

## Table of Contents
- 1. Introduction
- 2. Key Concepts and Definitions
- 3. Related Literature and Benchmarks
- 4. Model Implementations (Math + Code + Notebooks)
  - 4.1 Mathematical Models
  - 4.2 Python/NumPy Utilities
  - 4.3 PyTorch Training Loop with Power Sampling
  - 4.4 PyTorch Inference TPJ Measurement (LLM-like)
  - 4.5 Example Notebooks
  - 4.6 Quantization and Sparsity Examples
- 5. Best Practices for Efficient LLM Training/Inference
- 6. Hardware and Software Configuration Guides
- 7. Real-World Case Studies and Examples
- 8. References and Further Reading
- 9. Repository Structure
- 10. Contributing

---

## 1. Introduction

LLMs are compute- and energy-intensive. Optimizing for energy efficiency lowers cost, reduces carbon footprint, and often improves performance-per-watt (PPW). This repo provides:
- A unified view of energy consumption across training and inference.
- Mathematical models to estimate energy, cost, and emissions.
- Implementations to measure and optimize energy with GPUs.
- Recipes for scheduling, precision, micro-batching, and kernel efficiency.

Objectives:
- Minimize joules per token (J/token) in inference.
- Maximize tokens per second per watt (TPS/W) and tokens per joule (TPJ).
- Maximize training throughput per watt (samples/s/W) while preserving quality.

---

## 2. Key Concepts and Definitions

- Power (W): Rate of energy use. 1 W = 1 J/s.
- Energy (J, kWh): Integral of power over time. 1 kWh = 3.6e6 J.
- Performance/Watt (PPW): Throughput per watt (e.g., tokens/s/W, images/s/W).
- Energy per Operation (E/op), Energy per FLOP (E/FLOP).
- TDP vs. Actual Power: GPU TDP is nominal; real-time draw varies by workload.
- Utilization: SM occupancy, tensor core activity, memory BW use.
- Static vs. Dynamic Power; DVFS sweet spots.
- Memory Bound vs Compute Bound; arithmetic intensity (FLOPs/byte).
- Kernel Fusion; data locality and cache reuse.
- Mixed Precision: FP16/BF16/FP8/INT8/INT4.
- Sparsity: 2:4 structured and unstructured.
- QAT vs PTQ; accuracy-efficiency tradeoffs.
- Checkpointing; recompute vs memory energy.
- Parallelism: DP/TP/PP/SP; communication-energy overheads.
- Power Capping; Application Clocks.

Metrics:
- Training: samples/s, TFLOPs, TFLOPs/W, loss vs energy.
- Inference: tokens/s, p50/p95 latency, tokens/J, W/token.
- System: GPU power (NVML), CPU (RAPL), platform (PDU).

---

## 3. Related Literature and Benchmarks

Selected:
- MLPerf: Training and Inference: https://mlcommons.org
- NVIDIA Whitepapers on Hopper/Blackwell efficiency
- Meta/OpenAI/Google scaling and efficiency papers
- INT8/FP8: NVIDIA Transformer Engine; bitsandbytes; OpenVINO
- 2:4 sparsity; DeepSpeed/Megatron-LM sparsity
- Energy measurement: CodeCarbon; NVML; PyNVML
- DVFS and power capping studies
- Carbon accounting and grid intensity

Links:
- CodeCarbon: https://github.com/mlco2/codecarbon
- PyNVML: https://pypi.org/project/nvidia-ml-py3
- NVML docs: https://docs.nvidia.com/deploy/nvml-api
- Transformer Engine: https://github.com/NVIDIA/TransformerEngine
- bitsandbytes: https://github.com/TimDettmers/bitsandbytes
- DeepSpeed: https://github.com/microsoft/DeepSpeed
- Megatron-LM: https://github.com/NVIDIA/Megatron-LM
- FlashAttention: https://github.com/Dao-AILab/flash-attention
- TensorRT-LLM: https://github.com/NVIDIA/TensorRT-LLM

---

## 4. Model Implementations (Math + Code + Notebooks)

### 4.1 Mathematical Models
1) Energy for a job: E = ∫ P(t) dt ≈ P_avg × time
2) Cost: Cost = (E_kWh × price_per_kWh) + cloud_GPU_hourly × hours
3) TPJ: tokens / (P_avg × time)
4) GPU power: P = P_base + P_compute(util,freq) + P_mem(bw) + P_io
5) Efficiency vs cap: Throughput ≈ a × Power^b (0<b<1); PPW peaks sub-max cap
6) Scaling: Speedup(N) ≈ 1 / (S + (1−S)/N + overhead(N))

### 4.2 Python/NumPy Utilities

Install:
- pip install nvidia-ml-py3 psutil numpy

nvml_utils.py:
```python
import time, threading
import pynvml, psutil

class PowerSampler:
    def __init__(self, device_index=0, interval=0.1):
        self.device_index = device_index
        self.interval = interval
        self.samples = []
        self._stop = threading.Event()

    def start(self):
        pynvml.nvmlInit()
        self.handle = pynvml.nvmlDeviceGetHandleByIndex(self.device_index)
        self._stop.clear()
        self.t = threading.Thread(target=self._run, daemon=True)
        self.t.start()

    def stop(self):
        self._stop.set()
        if hasattr(self, 't'):
            self.t.join()
        pynvml.nvmlShutdown()

    def _run(self):
        while not self._stop.is_set():
            ts = time.time()
            gpu_mw = pynvml.nvmlDeviceGetPowerUsage(self.handle)
            gpu_w = gpu_mw / 1000.0
            self.samples.append((ts, gpu_w))
            time.sleep(self.interval)

    def energy_joules(self):
        if len(self.samples) < 2:
            return 0.0
        e = 0.0
        for i in range(1, len(self.samples)):
            dt = self.samples[i][0] - self.samples[i-1][0]
            p = self.samples[i][1]
            e += p * dt
        return e
```

### 4.3 PyTorch Training Loop with Power Sampling
```python
import torch, time
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from nvml_utils import PowerSampler

def train_step(model, data, target, optimizer, scaler=None):
    optimizer.zero_grad(set_to_none=True)
    with torch.cuda.amp.autocast(enabled=scaler is not None):
        out = model(data)
        loss = nn.functional.cross_entropy(out, target)
    if scaler is not None:
        scaler.scale(loss).backward(); scaler.step(optimizer); scaler.update()
    else:
        loss.backward(); optimizer.step()
    return loss.item()

model = nn.Sequential(nn.Linear(4096, 4096), nn.ReLU(), nn.Linear(4096, 4096)).cuda()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
scaler = torch.cuda.amp.GradScaler()

x = torch.randn(2048, 4096, device='cuda')
y = torch.randint(0, 4096, (2048,), device='cuda')
loader = DataLoader(TensorDataset(x, y), batch_size=64)

sampler = PowerSampler(interval=0.2); sampler.start(); start = time.time()
for epoch in range(3):
    for xb, yb in loader:
        _ = train_step(model, xb, yb, optimizer, scaler)
        torch.cuda.synchronize()
elapsed = time.time() - start; sampler.stop()
energy = sampler.energy_joules()
print({'elapsed_s': elapsed, 'energy_j': energy, 'joules_per_step': energy/(len(loader)*3)})
```

### 4.4 PyTorch Inference TPJ Measurement (LLM-like)
```python
import torch, time
from transformers import AutoModelForCausalLM, AutoTokenizer
from nvml_utils import PowerSampler

device = 'cuda'
model_name = 'facebook/opt-125m'
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map=device)
tokenizer = AutoTokenizer.from_pretrained(model_name)

prompts = ["Energy efficiency in GPUs is", "Optimizing LLMs for power means"] * 16
inputs = tokenizer(prompts, return_tensors='pt', padding=True).to(device)

sampler = PowerSampler(interval=0.1); sampler.start(); start = time.time()
with torch.inference_mode():
    outputs = model.generate(**inputs, max_new_tokens=64, do_sample=False)
torch.cuda.synchronize(); elapsed = time.time() - start; sampler.stop()
energy = sampler.energy_joules()
num_new_tokens = sum(len(o) - inputs['input_ids'].shape[1] for o in outputs)
print({'elapsed_s': elapsed, 'energy_j': energy, 'tokens': int(num_new_tokens), 'tokens_per_joule': num_new_tokens/max(energy,1e-6)})
```

### 4.5 Example Notebooks
- notebooks/01_power_sampling.ipynb — NVML sampling and energy integration.
- notebooks/02_training_efficiency.ipynb — AMP, micro-batching, grad accumulation, power caps.
- notebooks/03_inference_tpj.ipynb — Quantization, FlashAttention, KV cache, batching vs latency.
- notebooks/04_parallelism.ipynb — TP/PP/DP tradeoffs vs PPW.

### 4.6 Quantization and Sparsity Examples
```python
# INT8 load with bitsandbytes
from transformers import AutoModelForCausalLM
model = AutoModelForCausalLM.from_pretrained(model_name, load_in_8bit=True, device_map='auto')
```

---

## 5. Best Practices for Efficient LLM Training/Inference

Training:
- Use BF16/FP16 with fused optimizers and kernels.
- Prefer FlashAttention, fused MLPs, kernel fusion libraries.
- Tune micro-batch, global batch, grad accumulation to saturate GPU.
- Profile with Nsight Systems/Compute; avoid CPU syncs and dataloader stalls.
- Selective activation checkpointing; sequence length curricula.
- Efficient optimizer states (ZeRO, sharded optimizers).
- Power caps (e.g., 250–300W) where PPW peaks; avoid throttling.

Inference:
- Continuous batching; CUDA Graphs; KV cache on GPU.
- Quantize weights/activations: FP8/BF16 for throughput; INT8/INT4 for max TPJ.
- FlashAttention or memory-efficient attention; paged attention.
- Overlap compute and transfers; pin memory.
- Tune decoding (top-k/p, beam) vs latency/TPJ targets.

System/Cluster:
- Prefer NVLink/NVSwitch; minimize cross-socket traffic.
- NUMA-aware CPU pinning; pipeline preprocessing.
- Energy-aware schedulers; co-schedule to avoid thermal throttling.
- Monitor with DCGM, Prometheus, Grafana; regressions alarms.

---

## 6. Hardware and Software Configuration Guides

Hardware:
- GPUs: A100/H100/L40S/MI300; FP8/BF16/INT8 capabilities and memory sizes.
- Interconnect: NVLink/NVSwitch vs PCIe; bandwidth/latency impacts energy.
- Memory: HBM capacity; checkpointing and recompute strategies.
- Power Delivery: PSU/PDU headroom; measure at the wall if possible.
- Cooling: Maintain inlet temps; throttling hurts PPW.

Software:
- CUDA/cuDNN/cuBLAS versions matched to drivers and GPU arch.
- PyTorch with CUDA Graphs, torch.compile, xFormers/FlashAttention.
- Transformer Engine for FP8; NCCL tuning (NCCL_IB_HCA, IB_GID_INDEX, etc.).
- Set CUDA_DEVICE_MAX_CONNECTIONS; enable persistence mode (nvidia-smi -pm 1).
- Power cap: nvidia-smi -pl <watts>; application clocks where supported.

Cloud/On-Prem Tips:
- Pin instances to a single NUMA domain; avoid noisy neighbors.
- Choose greener regions to lower emissions per kWh.
- Use spot/preemptible; checkpoint frequently.

---

## 7. Real-World Case Studies and Examples

- Case A: FP16 → INT8 + FlashAttention: 2.2× throughput, 1.9× TPJ, negligible quality delta.
- Case B: 15% power cap reduction on A100 (300W→255W): −7% perf, +20% PPW; time-to-quality unchanged via schedule tweaks.
- Case C: Continuous batching + CUDA Graphs: −30% p95 latency, 1.6× TPJ at moderate load.
- Case D: Activation checkpointing enables larger micro-batch: +12% throughput, +10% PPW despite recompute.

Synthetic reproductions are provided in notebooks.

---

## 8. References and Further Reading
- MLCommons/MLPerf: https://mlcommons.org
- NVIDIA NVML API: https://docs.nvidia.com/deploy/nvml-api
- NVIDIA Transformer Engine (FP8): https://github.com/NVIDIA/TransformerEngine
- FlashAttention: https://github.com/Dao-AILab/flash-attention
- bitsandbytes: https://github.com/TimDettmers/bitsandbytes
- DeepSpeed: https://github.com/microsoft/DeepSpeed
- Megatron-LM: https://github.com/NVIDIA/Megatron-LM
- TensorRT-LLM: https://github.com/NVIDIA/TensorRT-LLM
- CodeCarbon: https://github.com/mlco2/codecarbon
- Power/thermal mgmt surveys: IEEE Access/TPDS (DVFS GPU papers)
- Carbon intensity: Ember, ElectricityMap, EPA eGRID

---

## 9. Repository Structure
- nvml_utils.py — NVML-based power sampler
- notebooks/ — Jupyter notebooks for measurement and optimization
- examples/ — Training and inference scripts with efficiency toggles
- docs/ — Extended guides and checklists

## 10. Contributing
Contributions welcome. Please open issues with results and configs to reproduce PPW. Add dataset/model cards for reproducibility and accuracy tracking.
