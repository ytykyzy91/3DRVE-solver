基于Python创建一个有限元求解器项目,用于复合材料RVE(代表性体积单元)的均质化分析。

## 核心功能
1. 有限元框架:支持读取VTU网格(四面体/六面体单元,具体单元类型请先解析example/case0001_fix.vtu确认)
2. 均质化方法:基于变分渐近法理论(Variational Asymptotic Method),通过求解6个独立的单胞边值问题获得等效刚度矩阵
3. 输出:均质化后的6x6刚度矩阵(Voigt记法),保存为json和csv两种格式
4. 边界条件:实现周期性边界条件(PBC),基于6个宏观应变分量(εxx, εyy, εzz, γxy, γyz, γxz)分别加载,预留接口供后续调用
5. 非线性接口:预留材料非线性(如塑性/损伤)的求解器接口,当前版本仅实现线弹性

## 输入文件
- 网格文件: example/case0001_fix.vtu (包含节点坐标、单元连接关系、单元Material ID)
- 材料配置: example/user_RVE_analysis.json

## 材料映射逻辑
VTU中Material ID与JSON材料定义的映射关系:
- Material[0] -> Defs.Composite.Materials.reinforcement
- Material[1] -> Defs.Composite.Materials.matrix  
- Material[2] -> Defs.Composite.Materials.Interphase

每个Materials.*条目包含一个material字段(材料名称),需以此为key在Defs.Analysis.Materials中查找对应的E(弹性模量)和nu(泊松比),用于构造各相材料的弹性张量。

## 第一步
请先解析两个输入文件的实际结构(VTU的单元类型/节点数/Material数组位置,JSON的完整字段层级),确认理解无误后再开始搭建项目框架。

## 技术要求
- 使用稀疏矩阵求解线性方程组(scipy.sparse)
- 代码结构模块化:网格读取/材料映射/单元刚度计算/全局组装/边界条件处理/求解/后处理 分文件实现
- 提供单元测试或简单验证案例(如均匀材料退化为各向同性验证)

我现在需要你实现基于宏观应变的应力应变分析，包括
1. 宏观应变的输入入口（可以是json配置文件）
2. 载荷步数量定义（用于非线性分析）
3. 输出vtu结果文件，在计算用输入vtu的基础上，增加displacement、mises stress、stress、strain等计算结果




 现在开始拓展非线性能力。
  - 只有基体存在非线性，描述形式参考【要求】
  - 基体的非线性参数单独放在一个文件，不要侵入已有的求解json
  - 注意求解的收敛性
  - 提供开关，可以选择是否进行非线性分析。

  【】
  纤维被视为各向同性线弹性材料，忽略纤维损伤。基体行为采用扩展线性 Drucker-Prager [28]
  准则进行建模，以考虑聚合物对静水应力的敏感性，从而在复杂载荷条件下提供准确的表征。该准则的数学表达式为：

  $$
  F = \tau - p \tan \beta - d = 0 \tag{2}
  $$

  其中 $\tau$ 代表屈服应力；$p = -\text{trace}(\sigma)/3$ 是静水应力；$\beta$ 是材料的摩擦角，决定了 $p - \tau$
  应力平面中屈服面的斜率，而 $d$ 是材料的粘聚力，代表在没有外部剪应力的情况下材料抵抗剪切破坏的内部阻力。


  为了模拟损伤的萌生和演化，本研究采用延性断裂准则结合基于能量的渐进损伤模型。当等效塑性应变达到临界值（记为 $\varepsil
  on_0^{pl}$）时，损伤开始萌生。随着塑性变形的进行，损伤不断演化，直到材料达到完全破坏，其特征为最终的等效塑性应变值
  $\varepsilon_f^{pl}$。损伤演化过程由裂纹扩展过程中耗散的能量控制，能量释放率（或单位断裂能）由以下积分给出：

  $$
  G_m = \int_{\varepsilon_0^{pl}}^{\varepsilon_f^{pl}} L \sigma_y d\varepsilon_{pl} \tag{4}
  $$

  其中 $L$ 是特征长度，$\sigma_y$ 是屈服应力，该积分捕获了材料从弹性变形过渡到塑性变形并最终断裂过程中耗散的能量。