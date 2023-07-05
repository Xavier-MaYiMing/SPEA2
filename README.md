### SPEA2

##### Reference: Zitzler E, Laumanns M, Thiele L. SPEA2: Improving the strength Pareto evolutionary algorithm[J]. TIK-Report, 2001, 103.

Strength Pareto Evolutionary algorithm 2 (SPEA2) with simulated binary crossover (SBX), polynomial mutation, and tournament selection.

SPEA2 belongs to the category of multi-objective evolutionary algorithms (MOEAs).

| Variables | Meaning                                                  |
| --------- | -------------------------------------------------------- |
| npop      | Population size                                          |
| narch     | Archive size                                             |
| iter      | Iteration number                                         |
| lb        | Lower bound                                              |
| ub        | Upper bound                                              |
| pc        | Crossover probability (default = 0.9)                    |
| eta_c     | The spread factor distribution index (default = 20)      |
| eta_m     | The perturbance factor distribution index (default = 20) |
| t         | Iterator                                                 |
| dim       | Dimension                                                |
| pop       | Population                                               |
| objs      | Objectives                                               |
| arch      | Archive                                                  |
| arch_objs | The objectives of archive                                |
| nall      | The total number of individuals                          |
| K         | The K-th nearest neighbor                                |
| S         | The strength value                                       |
| R         | The raw fitness                                          |
| dom       | Domination matrix                                        |
| sigma     | The distance between objectives                          |
| sigma_K   | The K-th shortest distance                               |
| D         | Density                                                  |
| F         | Fitness                                                  |

#### Test problem: ZDT3



$$
\left\{
\begin{aligned}
&f_1(x)=x_1\\
&f_2(x)=g(x)\left[1-\sqrt{x_1/g(x)}-\frac{x_1}{g(x)}\sin(10\pi x_1)\right]\\
&f_3(x)=1+9\left(\sum_{i=2}^nx_i\right)/(n-1)\\
&x_i\in[0, 1], \qquad i=1,\cdots,n
\end{aligned}
\right.
$$



#### Example

```python
if __name__ == '__main__':
    main(100, 100, 300, np.array([0] * 10), np.array([1] * 10))
```

##### Output:

![](https://github.com/Xavier-MaYiMing/SPEA2/blob/main/Pareto%20front.png)

