# 远程 GPU 跑实验 Runbook(RTX 5090)

面向**全新租用的 5090 服务器**,从零到产出实验 A/B/连接结果。重点:把不吃 GPU 的活(下载、调试)前置,GPU 只开必要的几十分钟。

---

## ⚠️ 头号坑:5090 是 Blackwell(sm_120),需 CUDA 12.8+ 的 torch

很多稳定版 PyTorch 没编 Blackwell 内核,会报 `no kernel image is available`。**第一件事**:

```bash
pip install --pre torch torchvision --index-url https://download.pytorch.org/whl/cu128
python -c "import torch; print(torch.__version__, torch.cuda.get_device_name(0), torch.cuda.is_available())"
# 期望: 2.x.dev...+cu128   NVIDIA GeForce RTX 5090   True
```

如镜像自带 torch,先验证上面这行;不行就按此重装(**且要在装其余依赖之前**,避免被覆盖成 CPU 版)。

---

## 📥 下载清单

| 项目 | 来源 | 大小 | 吃 GPU? |
|---|---|---|---|
| torch/torchvision (cu128) | pytorch.org cu128 wheel | ~2GB | 否 |
| Python 包 | `pip install numpy scipy scikit-learn timm pyyaml tqdm matplotlib seaborn` | 小 | 否 |
| I-JEPA 权重 (ViT-L/16) | `facebookresearch/ijepa` release 或 HF 镜像 | ~1.2GB | 否 |
| MAE 权重 (ViT-L/16) | `facebookresearch/mae`:`mae_pretrain_vit_large.pth` | ~1.2GB | 否 |
| SimCLR ResNet-18 | 社区 PyTorch SimCLR repo(优先,免 TF 转换) | ~45MB | 否 |
| ResNet-18 监督版 | torchvision 自动 | ~45MB | 否(需联网) |
| STL-10 | torchvision 自动(首次 `01` 触发) | ~2.5GB | 否 |
| CIFAR-10(交叉验证) | torchvision 自动 | ~170MB | 否 |
| ModelNet40 多视角(仅 Part C) | MVCNN 渲染 PNG | 几 GB | 否 |

**放置路径**(默认见 `configs/encoders.yaml`):
```
weights/ijepa_vitl16.pth
weights/mae_vitl16.pth
weights/simclr_resnet18.pth
```

> ❗**ViT-S/16 下载不到**:MAE/I-JEPA 官方都没有 ViT-S。骨架已默认改为 **ViT-L/16**(两者都有官方权重)。唯一硬约束:**I-JEPA 与 MAE 同骨架**。要换 ViT-B/16 就同步改两个 block 的 `arch`/`feature_dim`(见 encoders.yaml 注释)。

---

## ⚙️ 配置清单

`configs/encoders.yaml` —— 已默认 ViT-L/16,只需确认 checkpoint 路径与实际文件名一致。
若下到的是别的规格,**两个 block 一起改**:
| 规格 | arch | feature_dim |
|---|---|---|
| ViT-S/16 | `vit_small_patch16_224` | 384 |
| ViT-B/16 | `vit_base_patch16_224` | 768 |
| ViT-L/16 | `vit_large_patch16_224` | 1024 |

`configs/default.yaml` —— 通常只确认:
```yaml
device: cuda
features:
  batch_size: 256        # 5090 32GB 可调到 512
  random_projection:
    target_dim: 384      # 保持 384:把 1024/512 都投到同一维,正是"统一维度"的意义
paths:
  data_root: ./data      # 确认磁盘 ≥ ~6GB
```

**验证配置(不吃 GPU):**
```bash
pytest -q                                 # 期望 15 passed
python scripts/00_download_weights.py     # 必须打印 All encoders loaded successfully.
```
> 若报 missing/unexpected keys → 按 [工程推进计划.md](工程推进计划.md) 阶段 1.3 dump 真实键名,修 `vit.py`/`resnet.py` 的重映射函数。**这一步是最大变数,务必在便宜机器/本地先搞定。**

---

## ▶️ 执行顺序

```bash
# 冒烟:单 encoder 小批量,验证全链路(几分钟)
python scripts/01_extract_features.py encoders=[resnet18_supervised] features.batch_size=64
python scripts/02_svd_analysis.py     encoders=[resnet18_supervised]

# 实验 A 全量:抽特征(GPU)→ SVD/Two-NN
python scripts/01_extract_features.py     # 8 个缓存 .pt(4 encoder × train/test)
python scripts/02_svd_analysis.py         # Part A

# 顺手 B + 连接(复用缓存,几乎零 GPU)
python scripts/03_svm_probe.py            # Part B
python scripts/04_connect_analysis.py     # rank↔margin + 受控对照差值

# 取回结果(很小)
rsync -avz <server>:~/<repo>/results ./
```

**Gate A 验收**:`results/stl10/svd/` 有 `svd_summary.json` / `singular_spectra.png`;RankMe ∈ (1, 上限];看 I-JEPA vs MAE 的 RankMe 与 `curvature_gap`。

---

## ⏱️ 时间估算(5090,ViT-L/16,AMP)

| 阶段 | 纯计算 | 墙钟(含下载/调试) | GPU? |
|---|---|---|---|
| 环境(含 Blackwell torch) | — | 0.5–2h ⚠️ | 否 |
| 权重下载 | — | 0.2–0.5h | 否 |
| checkpoint 键名调试 | — | 0.5–3h 🎲 | 否 |
| 冒烟测试 | <2min | <5min | 轻 |
| **实验 A:抽特征(4 enc)** | 3–8min | +STL-10 下载 5–15min | **是** |
| **实验 A:SVD+Two-NN** | 2–5min | ~5min | 轻 |
| 实验 B:线性 SVM | 5–15min | ~15min | 否(CPU) |
| 连接分析 | <10s | <1min | 否 |
| CIFAR 交叉验证(全套) | +15–30min | — | 部分 |
| Part C(t-SNE 为主) | +1–2h | — | 否(t-SNE 不吃 GPU) |
| Part D(扰动) | +10–20min | — | 是(很快) |

**结论**
| 场景 | 总墙钟 | 建议 GPU 开机时长 |
|---|---|---|
| 核心(A+B+连接,STL-10) | 顺利 1.5–3h / 卡 checkpoint 5h+ | **~0.5–1h** |
| 核心 + CIFAR | 2–4h | ~1–1.5h |
| 全套(+C/D) | 4–7h | ~2–3h |

---

## 💰 省 GPU 钱三原则
1. **环境/下权重/调键名全在本地或便宜机器做完**(`00` 打印成功之前不需要 GPU)。
2. **特征只抽一次**,A/B/连接/Part D 全复用 `features/cache/`,别重复 `01`。
3. **Part C 的 t-SNE 不吃 GPU**:抽完特征即可关 GPU,把缓存拉到便宜机器慢慢画。
