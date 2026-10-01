# Logical Coset Distinguishability (LCD)

**表面码噪声的资源理论:噪声预序、局域汇率、解码器余量与有限数据外推认证**

研究计划全文见 [`ABSTRACT.md`](ABSTRACT.md)。本 README 同时是代码实现的任务说明,供开发者(包括 Claude Code)直接据此开工。

---

## 1. 核心思想(一段话)

在最大似然(陪集)解码下,逻辑失败概率等于"给定综合征、区分逻辑陪集"这一假设检验问题的贝叶斯错误率。因此,**逻辑陪集的可区分性**可以作为资源;综合征粗粒化、丢弃经典边信息、叠加可经典采样的独立噪声是自由操作,资源在其下不增。错误压制指数

```
α = lim_{d→∞} −(1/d) · log p_L(d)
```

是该区分问题的大偏差错误指数,也等于统计力学映射中畴壁的自由能张力。本仓库的数值部分用来验证并量化这一框架的推论。

## 2. 当前目标:两个初步结果

研究计划已基本定稿,当前最重要的是产出**具体数字**。按优先级:

### P1:码容量噪声下,擦除与泡利噪声的局域汇率

**设定**
- 旋转表面码,码距 d ∈ {5, 7, 9, 11, 13, 15}(张量网络能承受的话继续加大)。
- 码容量噪声:每个数据比特独立地,以概率 e 被擦除(施加均匀随机泡利 I/X/Y/Z,位置已知);否则以概率 p 发生去极化错误。
- 两种解码器:
  - **ML**:张量网络最大似然(陪集)解码,即 Bravyi–Suchara–Vargo 方法。擦除比特的先验设为 I/X/Y/Z 各 1/4,其他比特按去极化先验。可先尝试 `qecsim` 的 `PlanarMPSDecoder`,如不支持逐比特先验则自行实现。
  - **MWPM**:PyMatching,擦除比特边权设为 0(或极小)。

**要算的量**
1. 在 (p, e) 网格上估计 p_L(d; p, e),附 Wilson 置信区间。
2. 对每个 (p, e),拟合 `log p_L(d) ≈ a − α·d`,得到 α(p, e)。报告拟合所用 d 范围,并检查去掉最小码距后 α 是否稳定(有限尺寸修正的诊断)。
3. **局域汇率**:在工作点 (p₀, e₀) 处

   ```
   R_{e→p}(p₀, e₀) = −(∂α/∂e) / (∂α/∂p)
   ```

   含义:沿等 α 曲线,增加一个单位的擦除率,等价于增加多少泡利错误率。用有限差分或等 α 曲线斜率计算,并给出误差估计。
4. **解码器余量**:α_ML(p, e) − α_MWPM(p, e)。

**工作点建议**:e₀ ∈ {0, 0.02, 0.05},p₀ ∈ {0.02, 0.04, 0.06}。具体视统计量和计算时间调整。

**合理性检验(必须先通过)**
- 纯去极化噪声,ML 解码阈值约 18.9%。
- 纯擦除噪声,阈值约 50%。
- 纯比特翻转噪声,MWPM 阈值约 10.3%。

数值结果与上述已知值偏差明显时,先排查实现,不得继续往下做。

### P2:局域突发事件的检测,时空模式区分与泊松计数的对比

**设定**
- 用 stim 生成旋转表面码存储实验(memory experiment),电路级均匀去极化噪声 p = 1e-3 作为背景,码距 d ∈ {5, 7, 9},轮数可调。
- **突发事件注入**:以率 r(每比特每轮)在随机时空位置触发突发事件,事件影响半径 R 内的比特、持续 T_b 轮,期间这些位置的错误率升到 p_b(例如 0.1 到 0.3)。

  实现提示:对泡利噪声,检测事件对错误是线性的(综合征是泡利帧的奇偶校验)。因此可以分别采样"背景"和"仅突发区域有噪声的电路"所产生的检测事件,再按位异或叠加,不必在单个 stim 电路里表达随机的空间关联。

**对比两种方法**
- **基准(泊松计数)**:只统计逻辑失败次数或"高权重综合征"次数,据此检验 H₀(无突发)与 H₁(有突发,率 r)。
- **模式方法**:利用检测事件在时空上的聚集特征,例如滑动窗口内的局部检测事件计数、似然比或匹配滤波统计量,做同一检验,并估计 r 和 R。

**要报告的量**
- 在给定显著性 δ 和检验功效下,两种方法所需的轮数(或 shots),以及二者的比值。
- r 与 R 的估计偏差和方差。
- 明确对照泊松极限 ln(1/δ)/r:模式方法的优势体现在常数和对事件类别的区分上,**不应声称突破 1/r 标度**。

## 3. 仓库结构(建议)

```
.
├── ABSTRACT.md
├── README.md
├── pyproject.toml
├── src/lcd/
│   ├── noise/              # 噪声模型:去极化、擦除、局域突发、芯片尺度事件
│   ├── codes/              # 旋转表面码的构造、校验矩阵、逻辑算符
│   ├── decoders/
│   │   ├── tn_ml.py        # 张量网络最大似然解码(码容量)
│   │   └── mwpm.py         # PyMatching 封装,支持擦除
│   ├── circuits/           # stim 电路生成与突发事件注入
│   └── analysis/
│       ├── fit_alpha.py    # α 拟合与有限尺寸诊断
│       ├── exchange.py     # 局域汇率
│       └── detection.py    # 泊松基准与模式方法的检验统计量
├── experiments/
│   ├── p1_erasure_pauli/   # 运行脚本与配置
│   └── p2_burst_detection/
├── results/                # 原始数据(CSV)与图
└── tests/                  # 单元测试与阈值合理性检验
```

## 4. 环境

Python ≥ 3.10。

```bash
pip install stim sinter pymatching numpy scipy matplotlib pandas
pip install qecsim        # 可选,用于张量网络 ML 解码的对照
pip install quimb         # 可选,自行实现张量网络时使用
```

## 5. 工程规范

- **可复现**:所有随机过程使用显式种子;每次运行把配置(码距、噪声参数、shots、种子、代码版本)和结果一起写入 `results/`。
- **统计**:每个 p_L 都附置信区间;拟合给出参数误差。失败次数少于约 100 的点要在图中标注。
- **测试先行**:第 2 节的阈值合理性检验写成 `tests/` 中的测试,通过后再跑正式实验。
- **数据采集**:电路级实验优先用 `sinter` 并行采集。
- **不过度声明**:报告中区分"严格结论"和"数值估计";MWPM 结果不得表述为 ML 结果;码容量结论不得外推到电路级噪声。

## 6. 里程碑

| 阶段 | 内容 | 验收标准 |
|---|---|---|
| M0 | 码构造、噪声模型、MWPM 解码、合理性检验 | 三项阈值检验通过 |
| M1 | 张量网络 ML 解码(码容量,含擦除先验) | 去极化 ML 阈值约 18.9%,与 MWPM 结果趋势一致 |
| M2 | P1 完整结果 | 在至少 3 个工作点给出 R_{e→p} 和 ML−MWPM 余量,附误差 |
| M3 | 突发事件注入与泊松基准 | 注入的事件率能被无偏估计 |
| M4 | P2 完整结果 | 给出模式方法相对泊松计数所需实验时长的比值,附误差 |
| M5 | 汇总 | `results/SUMMARY.md`:关键数字、图、局限 |

## 7. 范围之外(当前阶段不做)

- 相干误差(框架依赖旋转化近似,见 ABSTRACT 的局限部分)。
- qLDPC 码。
- 电路级噪声下的精确 ML 解码。
- 理论证明(预序定理、认证定理)。本仓库只做数值部分,理论推导另行整理。

## 8. 参考文献

- E. Dennis, A. Kitaev, A. Landahl, J. Preskill, *Topological quantum memory*, J. Math. Phys. (2002).
- S. Bravyi, M. Suchara, A. Vargo, *Efficient algorithms for maximum likelihood decoding in the surface code*, Phys. Rev. A (2014).
- C. T. Chubb, S. T. Flammia, *Statistical mechanical models for quantum codes with correlated noise*, Ann. Inst. Henri Poincaré D (2021).
- C. Gidney, *Stim: a fast stabilizer circuit simulator*, Quantum (2021).
- O. Higgott, C. Gidney, *Sparse Blossom: correcting a million errors per core second with minimum-weight matching* (PyMatching v2).
- Y. Wu, S. Kolkowitz, S. Puri, J. D. Thompson, *Erasure conversion for fault-tolerant quantum computing in alkaline earth Rydberg atom arrays*, Nat. Commun. (2022).
- M. McEwen et al., *Resolving catastrophic error bursts from cosmic rays in large arrays of superconducting qubits*, Nat. Phys. (2022).
- Google Quantum AI, *Quantum error correction below the surface code threshold*, Nature (2025).

## 9. 许可证

待定。
