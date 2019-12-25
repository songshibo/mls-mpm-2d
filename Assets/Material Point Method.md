# Material Point Method

### Main Procedure

##### Quadratic B-spline weight

$$
\mathrm{N}(x)=\left\{\begin{array}{ll}{\frac{3}{4}-|x|^{2}} & {0 \leqslant|x|<\frac{1}{2}} \\ {\frac{1}{2}\left(\frac{3}{2}-|x|\right)^{2}} & {\frac{1}{2} \leqslant|x|<\frac{3}{2}} \\ {0} & {\frac{3}{2} \leqslant|x|}\end{array}\right.
$$

in actual code(discretely & calculated without judging input range)

for each cell in 9 surrounding cells $w_{ip} = w[x]_x * w[y]_y$， $x,y$ is in $\{0,1,2\}$ . so there has to be 6 value for $x$ and $y$. According to paper, $N()$ takes $x_p - x_i$ as input. And the cell center is defined as the center of the square, so the input will be $x_{diff} = x_p - (int2)x_p - (0.5, 0.5)$. 
$$
\begin{aligned}
w[0] = \frac{1}{2} * (\frac{1}{2} - x_{diff})^2 \\
w[1] = \frac{3}{4} - x_{diff}^2 \\
w[2] = \frac{1}{2} * (\frac{1}{2} + x_{diff})^2
\end{aligned}
$$
 $w[0],w[2]$ refer to $N(x)\{{-\frac{3}{2} < x \leq -\frac{1}{2}}\}$ and $N(x)\{\frac{1}{2} \leq x < \frac{3}{2}\}$. and both are shifted to $[0,1]$ and are symmetrical about $x = \frac{1}{2}$. 

##### P2G Mass Contribution

$$
m_i = \sum_{p} m_p * w_{ip}
$$

just loop through 9 surrounding cells and calculate each cell’s weight and sum together.

##### P2G Momentum Contribution

$$
m_p * C_p *(x_i - x_p) * w_{ip}
$$

$C_p$ is affine momentum matrix stored in each particle

###### $C_p$ update

$$
C_p^{n+1} = B * D^{-1} = \underbrace{\sum v_i * (x_i - x_p) * w_{ip}}_{B} * \underbrace{\frac{1}{\frac{1}{4} * {\Delta x}^2}}_{D^{-1}}
$$

in actual code, $\Delta x$ will be scaler(often 1, so $D^{-1}$ will be 4)

