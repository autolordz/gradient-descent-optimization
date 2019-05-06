# gradient-descent-optimization

a python script of a function summarize some popular methods about gradient descent  
损失函数为普通最小二乘OLS  
机器学习模型与优化求解的练手项目  
Python编写并使用简明的数学符号  

## TOC
<!-- MarkdownTOC -->

- [Updated](#updated)
- [Features](#features)
- [Usage](#usage)
- [Reference](#reference)
- [Licence](#licence)

<!-- /MarkdownTOC -->

<a id="updated"></a>
## Updated  
【2019-4-25】
添加约束优化L1L2等内容  
![](./img/gf4.png)

<a id="features"></a>
## Features

- [x] 一阶优化(非约束)
	- 原版
		- GD 全批梯度下降 mm10
		- minibatch-GD 小批梯度下降 mm10
		- SGD 随机梯度下降 mm10
	- 带动量
		- Polyak’s 动量 mm21,mm22
		- Nesterov 加速梯度 (NAG) mm23,mm24,mm25
		- FISTA (NAG 近端梯度版) mm26
- [x] 二阶优化
	- 牛顿法 mm30
	- 沿直线最小化 mm31
	- 共轭梯度法 mm32
	- 拟牛顿法Quasi-Newton(Broyden) mm33
- [x] 非凸优化
	- Adagrad(自适应梯度) mm40
	- RMSProp mm41
	- Adadelta mm42
	- Adam mm43
	- AdaMax mm44
	- Nadam mm45
	- AMSGrad mm46
- [x] 坐标下降
	- 循环坐标逐步 mm90
	- 循环坐标标准 mm91,mm92
	- 随机坐标 mm93
	- 随机块坐标 mm93
- [x] 速率
	- Armijo rules mm11
	- [ ] Wolfe conditions
- [x] 约束优化
	- L0(Best subset selection)
		- 向前逐步回归(匹配追踪)
	- L1(Lasso) 
		- 软阈值坐标下降
		- 近端梯度(广义梯度)
		- 对偶ADMM
	- L2(岭回归)
	- L2(组Lasso)
	- L1+L2(弹性网)
	- L1+L2(稀疏组Lasso)
- [x] 矩阵相关
	- [x]矩阵填充(分解)
		- 稀疏矩阵近似 ∥∥Z^∥∥l1 ≤ c
		- 奇异值分解 rank(Z^) ≤ c
		- 原子范数约束  ∥∥Z^∥∥⋆ ≤ c
		- 惩罚SVD Z^=UDVT,ϕ1(uj)≤c1,ϕ2(vk)≤c2
- [ ] 非凸约束
	- 自适应Lasso

<a id="usage"></a>
## Usage

spyder下直接块运行
或者py xxx.py

Sample:  
![](./img/gf1.png)
![](./img/gf2.png)
![](./img/gf3.png)

矩阵近似(填充)的推荐系统：
===svd aproxi===
====r= 2 =====
 0.47 %
 0.94 %
1 : obj= 0.04616 ratio= 0.7232133195138838
[[ 0.865  0.016  0.175    nan    nan]
 [-0.6      nan -0.212    nan    nan]
 [-0.717    nan    nan  0.064 -0.098]
 [ 0.697 -0.503  0.558  1.544    nan]
 [ 1.231 -0.342 -0.81  -0.82  -0.133]
 [ 0.266  0.145    nan    nan -2.241]]

 [[ 8.655e-01  1.565e-02  1.748e-01  1.223e-03 -2.939e-02]
 [-6.004e-01 -1.474e-01 -2.119e-01 -3.598e-02  1.956e-02]
 [-7.169e-01 -1.224e-01  2.289e-02  6.437e-02 -9.754e-02]
 [ 6.966e-01 -5.033e-01  5.585e-01  1.544e+00 -2.776e-01]
 [ 1.231e+00 -3.423e-01 -8.103e-01 -8.201e-01 -1.326e-01]
 [ 2.664e-01  1.449e-01 -3.540e-02  2.657e-01 -2.241e+00]]
===svd aproxi===
====r= 2 =====
 0.47 %
 0.94 %
1 : obj= 0.09323 ratio= 0.7295933604564577
[[ 0.865  0.016  0.175    nan    nan]
 [-0.6      nan -0.212    nan    nan]
 [-0.717    nan    nan  0.064 -0.098]
 [ 0.697 -0.503  0.558  1.544    nan]
 [ 1.231 -0.342 -0.81  -0.82  -0.133]
 [ 0.266  0.145    nan    nan -2.241]]

 [[ 0.865  0.016  0.175  0.129 -0.309]
 [-0.6    0.023 -0.212 -0.127  0.211]
 [-0.717  0.003  0.006  0.064 -0.098]
 [ 0.697 -0.503  0.558  1.544 -0.209]
 [ 1.231 -0.342 -0.81  -0.82  -0.133]
 [ 0.266  0.145 -0.099 -0.036 -2.241]]

<a id="reference"></a>
## Reference

[梯度下降总结](http://ruder.io/optimizing-gradient-descent/index.html#nesterovacceleratedgradient)  
[卡内基凸优化2012秋](https://www.cs.cmu.edu/~ggordon/10725-F12/schedule.html)  
[SGD wiki](https://en.wikipedia.org/wiki/Stochastic_gradient_descent)  
[Armijo wiki](https://en.wikipedia.org/wiki/Backtracking_line_search)  
[Wolfe wiki](https://en.wikipedia.org/wiki/Wolfe_conditions)  
[Quasi-Newton wiki](https://en.wikipedia.org/wiki/Quasi-Newton_method)  
[分布式机器学习:算法、理论与实践-第4、5章](https://item.jd.com/12444377.html)  
[深度学习核心技术与实践-第6章](https://item.jd.com/12316912.html)  
[神经网络设计-第9章](hagan.okstate.edu/nnd.html)  
[[稀疏统计学习及其应用](https://item.jd.com/12296738.html)statistical learning with sparsity](https://web.stanford.edu/~hastie/StatLearnSparsity_files/SLS_corrected_1.4.16.pdf)  
[坐标下降软阈值Lasso推导](https://stats.stackexchange.com/questions/123672/coordinate-descent-soft-thresholding-update-operator-for-lasso)  
[矩阵分析与应用-张贤达-第3、4章](https://book.douban.com/subject/1257113/)  

<a id="licence"></a>
## Licence

[MIT Go to](#gradient-descent-optimization)

THE END
Enjoy