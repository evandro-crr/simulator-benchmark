from math import pi, sqrt
from ket import quant, H, around, phase_on, measure
from functools import partial


def grover(n: int, oracle) -> int:
    s = H(quant(n))
    steps = int((pi/4)*sqrt(2**n))
    for _ in range(steps):
        oracle(s)
        with around(H, s):
            phase_on(0, s)
    return measure(s).value


grover_w3 = partial(grover, oracle=phase_on(3))
