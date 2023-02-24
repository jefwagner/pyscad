## Polynomials

$$
p(x) = \sum_{i=0}^n c_i \, x^i
$$
$$
p(x) = c_n \, x^n + c_{n-1} \, x^{n-1} + \cdots + c_0
$$
$$
p(x) = x\big(c_n \, x^{n-1} + c_{n-1} \, x^{n-2} + \cdots + c_1 \big) + c_0
$$
$$
p(x) = x\Big(x\big(c_n \, x^{n-2} + c_{n-1} \, x^{n-3} + \cdots + c_2
\big) + c_1 \Big) + c_0
$$

$$
p(x) = f_n(x) \quad\text{where}\quad
f_0(x) = c_n, \,\,\,\text{and}\,\,\,
f_i(x) = x\,f_{i-1}(x) + c_{n-i}.
$$

$$
p'(x) = f'_n(x)
$$
$$
f_i(x) = x \,f_{i-1}(x) + c_{n-i} \\
f'_i(x) = x \, f'_{i-1}(x)  + f_{i-1}(x) \\
f''_i(x) = x \, f''_{i-1}(x) + 2 f'_{i-1}(x) \\
\vdots \\
f^{(p)}_i(x) = x \, f^{(p)}_{i-1}(x) + p f^{(p-1)}_{i-1}(x) \\
$$ 