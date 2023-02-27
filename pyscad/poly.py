from typing import Sequence, Union
import numpy as np

class Poly:
    """A finite polynomial with floating point coefficients"""
    
    def __init__(self, coef: Sequence[float]):
        """Create a new polynomial object"""
        self.coef = np.trim_zeros(np.array(coef, dtype=np.float64), 'b')

    def __repr__(self) -> str:
        """Build a string representation of a polynomial"""
        output = f'{self.coef[0]:.02f}' if self.coef[0] != 0 else ''
        for i, c in enumerate(self.coef[1:]):
            if c != 0:
                plus = ' + ' if output != '' else ''
                power = f'^{i+1}' if i>0 else ''
                output += f'{plus}{c:.02f}*x{power}'
        return output

    def __call__(self, x: float) -> float:
        """Evaluate the polynomial"""
        f = np.full_like(x, self.coef[-1])
        for c in self.coef[-2::-1]:
            f = x*f + c
        return f

    def d(self, x: float, n : int = 1) -> float:
        """Evaluate the n^th derivative of the polynomial"""
        x = np.array(x)
        shape = list(x.shape) + [n+1]
        fv = np.zeros(shape, dtype=np.float64)
        # fv = np.zeros((n+1,), dtype=np.float64)
        fv.T[0] = self.coef[-1]
        nv = np.array(range(n+1))
        for c in self.coef[-2::-1]:
            adder = nv * np.roll(fv,1)
            adder.T[0] = c
            fv = (x*fv.T + adder.T).T
        return fv.T[-1]

    def __neg__(self) -> 'Poly':
        """Negate a polynomial"""
        return Poly(-self.coef)

    def __add__(self, other: 'Poly') -> 'Poly':
        """Add two polynomials"""
        ls = len(self.coef)
        lo = len(other.coef)
        if ls > lo:
            _coef = other.coef.copy()
            _coef.resize((ls,))
            _coef += self.coef
        elif lo > ls:
            _coef = self.coef.copy()
            _coef.resize((lo,))
            _coef += other.coef
        else:
            _coef = self.coef + other.coef
        return Poly(_coef)

    def __sub__(self, other: 'Poly') -> 'Poly':
        """Subtract two polynomials"""
        return self+(-other)

    def __rmul__(self, other: float) -> 'Poly':
        """Multiply a polynomial by a constant"""
        return Poly(other*self.coef)

    def polymul(self, other: 'Poly') -> 'Poly':
        """Multiple two polynomials"""
        return Poly(np.convolve(self.coef, other.coef))

    def __mul__(self, other: Union[float,'Poly']) -> 'Poly':
        """Multiply two polynomials together"""
        if isinstance(other, (float, int)):
            return other*self
        elif isinstance(other, Poly):
            return self.polymul(other)
