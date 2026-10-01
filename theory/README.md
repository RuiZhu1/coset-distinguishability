# theory/ —— 理论线

理论与数值**并行**推进,理论在前面带路,数值用来检验理论(包括单调性引理)。

```
theory/
├── main.tex                   主文件(XeLaTeX + ctex);`make pdf` 生成 main.pdf
├── sec1-framework.tex         线性解码结构、可模拟约化、三类自由操作的单调性
├── sec2-representation.tex    擦除/泄漏/突发/芯片事件在模型中的表示,论证的适用边界
├── sec3-exchange.tex          汇率上界、码容量泡利+擦除类的预序定理、缺口问题
├── sec4-certification.tex     条件认证:联合上界、参数置信集、地板;突发情形的开放引理
├── sec5-information.tex       Chernoff/联合夹逼、汇率参照值、泊松下限、检测指数
├── sec6-map.tex               理论—数值对照表、开放问题、与 M0–M5 的并行顺序
└── checks/exact_small_codes.py  精确(无采样、无近似)ML 检验,n ≤ 9 的三个码
```

## 构建与运行

```bash
make -C theory pdf      # 需要 TeX Live: xelatex, latexmk, ctex, 中文字体
make -C theory check    # 需要 numpy;约 35 秒;不等式被违反时退出码非零
```

`check` 输出末行应为 `ALL INEQUALITY CHECKS PASSED`。

## 状态标签

| 标签 | 含义 |
|---|---|
| `PROVED` | 笔记里给出完整论证。**仍是初稿,尚未经独立复核**;建议作者逐行检查后再对外引用。 |
| `SKETCH` | 论证框架完整,个别步骤引用标准结果(如 Chernoff 指数)或几何细节只作了概述。 |
| `NUMERICAL` | 由 `checks/` 中的精确计算观察到,尚无证明。 |
| `CONJECTURE` | 明确的猜想。 |
| `PLAN` | 只有陈述与证明路线。 |

## 状态看板

| 编号 | 内容 | 状态 | 数值检验 |
|---|---|---|---|
| 定理 1.7 | 可模拟约化下 $\varepsilon^\star$ 单调(粗粒化/丢弃标记:数据处理;叠加:模拟论证) | PROVED | C1, C2 |
| 命题 1.9 | (F1)(F2)(F2′) 是可模拟约化;叠加独立噪声不降低 $\varepsilon^\star$ | PROVED | C1, C2 |
| 命题 2.1 | 擦除 = 带标记的完全去极化(泡利扭曲);最大混态模型对真实噪声保守 | PROVED | — |
| 命题 2.7 | 假设 (L) 下泄漏的单调性;(L) 失效的位置 | PROVED(在假设下) | — |
| 命题 2.10 | 事件层速率方向的单调性(泊松叠加) | PROVED | — |
| 注 2.11 / 问题 4.9 (O1) | 形状参数 $(R,T_b,p_b)$ 的单调性:无标记时**不**由自由操作给出 | 开放 | — |
| 定理 3.1 | 部分丢弃标记不等式 $\varepsilon^\star(p,e_0{+}\delta)\le\varepsilon^\star(p'',e_0)$ | PROVED | C3 |
| 定理 3.2 | 汇率上界 $R_{e\to p}\le(3/4-p_0)/(1-e_0)$ | PROVED | C7;**P1(M2)** |
| 观察 3.5 | 上界对 $[[5,1,3]]$ 在 $e_0{=}0$ 处取等(8 位有效数字) | NUMERICAL | C7 |
| 定理 3.7 | 码容量泡利+擦除类上局域自由操作预序的充要条件 | PROVED | C4 |
| 开放问题 3.10 | 汇率缺口 $c-R_\alpha$ 来自码相关性还是自由操作不完备 | 开放 | **P1(M2)** |
| 注 3.13 / 观察 3.14 | ML 导数的包络论证(P1 采样方案用一张 f 表求 ∂p_L/∂p、∂p_L/∂e);小码上精确验证 | SKETCH / NUMERICAL | C8 |
| 猜想 3.11 | 码通用的操作序 = 局域自由序(充要条件的“强”形式) | CONJECTURE | 反例搜索(待做) |
| 猜想 3.12 | $H_B$:$\alpha$ 是 Bhattacharyya 参数 $B$ 的函数,$R_\alpha=R_B$ | CONJECTURE | **P1(M2)** |
| 引理 4.1 / 推论 4.2 | 联合(Bhattacharyya)上界 $\varepsilon^\star\le\tfrac12\widetilde W(B)$ | PROVED | C5 |
| 定理 4.5 | 码容量情形的有限样本认证(与 $\pL$ 数据无关) | PROVED | 覆盖率模拟(待做) |
| 引理 4.6 | 芯片尺度事件的地板 $\varepsilon^\star\ge q_c(1-2^{-q})$ 与上界 | PROVED | **P2(M3–M4)** |
| 引理 4.7 | 突发混合的联合上界 | PROVED | — |
| 定理 4.8 | 含突发+芯片事件的条件认证 | PLAN(缺 O1–O3) | P2 + 小 $d$ 精确 ML |
| 命题 5.1 | Chernoff 上界 $\alpha_+\le\ln(1/B)$ | SKETCH | C6($d{=}3$) |
| 命题 5.2 | 联合下界 $\alpha_-\ge-\Phi(B)$ | PROVED | — |
| 命题 5.5 | 泊松下限 $T\ge\ln((1-a)/\delta)/r$(模式方法不能突破 $1/r$) | PROVED | **P2(M4)** |
| 命题 5.6 | 类别区分的指数链 $D_{\rm count}\le D_{\rm pattern}\le D_{\rm event}$ | PROVED(不等式链) | **P2(M4)** |

编号以 `main.pdf` 中的实际编号为准(本表随笔记更新时核对)。

## 与数值线的两条硬约定

1. **每条定理配一个证伪测试。** 违反即说明定理或实现有错。测试放在 `checks/` 或 `tests/`。
2. **MWPM 不能证伪针对 ML 的上界。** 单调性、联合上界、认证定理都是对 Bayes 最优(ML)解码的陈述;
   MWPM 失败率 $\ge\varepsilon^\star$,既不能证实也不能证伪它们。必须用精确 ML(小码)或张量网络 ML。
   下界(地板、泊松下限、先知下界)对任何解码器成立,可以用 stim + MWPM 检验。

`checks/exact_small_codes.py` 也是 M1(张量网络 ML)的 oracle:$d=3$ 旋转表面码上 TN-ML 的 $\varepsilon^\star(p,e)$
必须与它在机器精度内一致。
