# The initial orbit determination(IOD) using short arc observation data from space-based telescopes
=============================================================

## *1. Laplace IOD*

对于空间目标初始轨道确定的过程，即为天基望远镜在对空间目标的可观测弧段内，利用短弧段的多组观测数据得到空间目标在某一时刻对应的位置和速度。

定初轨得到的位置和速度不仅可以在进一步精密定轨的过程中提供初始数据，也可以对再次捕获探测目标提供便利。因此，对于初定轨获得的轨道参数就有了一定的精度要求，不能偏离真实数据过远。本文是仅基于天基测角信息对空间目标进行定轨，所以选用了 Laplace 方法。


`r`是观测目标在地心坐标系下的位置，`R`是平台（天基望远镜）在地心坐标系下的位置，`rho`是平台相对目标的距离矢量:    

![1](http://latex.codecogs.com/svg.latex?\\vec{r}=\vec{R}+\vec{\rho})

设`ti`时刻时，空间目标的测量矢量在地心坐标系下的赤经和赤纬分别为`ai`，`bi`。`L`为观测的单位矢量，`Lx`,`Ly`,`Lz`为为该单位矢量的三分量。

![2](http://latex.codecogs.com/svg.latex?\\vec{\rho}=\rho\vec{L})

![3](http://latex.codecogs.com/svg.latex?Lx=cos(b)*cos(a))

![4](http://latex.codecogs.com/svg.latex?Ly=cos(b)*sin(a))

![5](http://latex.codecogs.com/svg.latex?Lz=sin(b))

由于运动方程可以展开成时间间隔  Δt = t-t0 的幂级数解

![6](http://latex.codecogs.com/svg.latex?\\vec{r(t)}=\vec{r0}+\vec{r0^{'}}\Delta+\frac{1}{2}\vec{r0^{''}}\Delta^{2}+\cdots)

所以，`t`时刻的位置矢量可以写成如下形式（本文算法中该方程仅基于二体方程，未考虑J2等其他摄动）：

![7](http://latex.codecogs.com/svg.latex?\\vec{rt}=f(\vec{r0},\vec{r0^{'}},\Delta)*\vec{r0}+g(\vec{r0},\vec{r0^{'}},\Delta)*\dot\vec{r0})

注：f和g的幂级数展开式可以通过力模型得到，该幂级数展开式中的部分项是由待求的r0，v0决定，所以在迭代过程中，f和g也不断变化

整理后可得如下条件方程：

![6](http://latex.codecogs.com/svg.latex?\\vec{L}\times\vec{R}=f(\vec{L}\times\vec{r0})+g(\vec{L}\times\dot\vec{r0}))

将其化为标量方程后可以看出方程个数与未知数个数并不一致，共有三个方程和六个未知数(r0,v0)，但三个方程只有两个是互相独立的，所以至少需要三次观测资料才能定轨

对于待求方程，只能采用迭代的方式求解，而且也可以通过迭代的方式求解 f，g

具体求解过程如下：

1. 通过坐标转换，计算平台坐标 Ri（Xi,Yi,Zi)
2. 计算观测向量的单位矢量（仿真中可以直接得到，不用再依赖测角转换）
3. 对于r0，选择一个合适的初值，并得出f, g的初始近似值（r, f，g初值一般取1,1,Δt）
4. 通过 f, g 的近似值计算r0，v0
5. 由新的r0，v0的值求得新的f和g的近似值
6. 通过迭代，多次计算r0，v0的近似值，直到满足一定要求为止
7. 根据r0，v0的值计算轨道六根数


本方法在初值选取，观测精度，摄动方程构建等许多方面都影响最后的精度，甚至导致无法求解

##  *2. code*

### 2.1 数据仿真

1. 天基望远镜和空间目标的轨道外推利用了`python`的`poliastro`构建轨道力模型（为了方便计算，摄动只考虑了J2项），并用该库的`cowell`算法进行轨道外推。

        初始状态的天基望远镜的六个轨道根数假设为: a = 6874897m, e = 0.001465, i = 98, w = 244, Ω= 46, v = 169


2. 空间目标的轨道仿真分为低轨和中高轨两种情况，其中，中高轨的半长轴在10000km到20000km以内取随机值，低轨的半长轴在6700km到7000km以内取随机值，其余五个根数都在合理范围内取随机值

        运行`space based short arc IOD high.ipynb`，即可得到多组中高轨空间目标的仿真数据，保存在`high_hypo_tmp_car.npy`，将其中弧段最长的仿真数据保存成适配接下来计算初轨的格式，即`obs_simu_high.txt`,即可进行下一步运算。
    
        同样，运行`space based short arc IOD low.ipynb`，即可得到多组低轨空间目标的仿真数据，保存在`low_hypo_tmp_car.npy`，将其中弧段最长的仿真数据保存成适配接下来计算初轨的格式，即`obs_simu_low.txt`,即可进行下一步运算。


3. 在天基望远镜对空间目标的观测中未考虑考虑地影、地球遮挡等无法观测情况，后续会构建更合理的观测模型。


4. 仿真的时间长度为6000s，即对于中低轨卫星和天基望远镜该仿真长度约为一个轨道周期左右。

### 2.2 初轨计算

1. 代码是将开源的基于地基的拉普拉斯算法改进为基于天基的拉普拉斯算法

        该代码的数据输入的格式为如下的多组观测数据：
        观测时刻 | 测量向量的方位角 高度角 | 天基望远镜的笛卡尔坐标系下的位置
        eg.
        2008| 1| 1| 0| 34| 8.0| 286.7697502504564| 9.649416183091473| -2679836.135372887| -6075477.248442299| -1785617.8571519025 

2. 修改`demo.py`中读入的文件名 ( 改为新数据的文件名 ) ,运行即可得到定轨结果。
3. 以下为部分中高轨空间目标的定轨结果：

        1.
        a_Laplace:  11255986.321064994             a_true:  11245477.960215928 
        ecc_Laplace:  0.013110727118079766         ecc_true:  0.012875088976667655
        inc_Laplace:  105.06768458398712           inc_true:  105.06746554138576
        raan_Laplace:  220.41562792122252          raan_true:  220.41562792122252
        w_Laplace:  167.48078222081523             w_true:  168.36067992043036
        M_Laplace:  172.50492554860475             M_true:  -67.91365282479835
        2.
        a_Laplace:  10160040.752461253             a_true:  10160262.359542023
        ecc_Laplace:  0.008846914410047231         ecc_true:  0.00895785191394856
        inc_Laplace:  57.68269479425257            inc_true:  57.645380092621906
        raan_Laplace:  252.4703907937993           raan_true:  252.43571528123564
        w_Laplace:  62.23765739683134              w_true:  59.14330134326914
        M_Laplace:  -88.50871602810626             M_true:  56.23123271496681

        总结：
        对中高轨多组定轨结果统计可以看出平近点角的相对误差较大，但半长轴、偏心率、轨道倾角、升交点黄经、近地点辐角相对误差结果较好，分别约为0.1%, 1.2%, 0.7%, 0.5%, 5.3%

4. 低轨空间目标的定轨结果绝大部分都无法收敛，出现第一种结果，一小部分收敛到平台轨道，出现第二种结果，与仿真的空间轨道误差较大：

        1.   
        a_Laplace:  -9921.220930540081
        ecc_Laplace:  1269.778926571368
        inc_Laplace:  55.05676950461502
        raan_Laplace:  12.930750188271912
        w_Laplace:  189.93702906271696
        M_Laplace:  0.8297492521179378
        2.
        a_Laplace:  6905106.794494414             
        ecc_Laplace:  0.003941343940117385        
        inc_Laplace:  97.9492511606192            
        raan_Laplace:  244.06544201080476         
        w_Laplace:  337.47120405092966        
        M_Laplace:  -13.407685636246923           

        总结：
        利用天基平台的纯角度的Laplace短弧定轨算法在大多数情况下无法收敛，或者会出现收敛到平台的情况。

        对于中高轨情况，算法可以得到较为准确的解，但对于低轨情况，算法无法收敛，目前采用一个周期内的多个弧段也无法收敛

5. 后续首先将在下一步通过仿真，结合地基的观测辅助天基定轨，其次很多论文也指出传统的定轨算法（eg. Laplace算法）并不适用于天基仅角度的短弧定轨，可能需要复现一些新的算法。
