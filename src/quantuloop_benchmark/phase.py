from math import pi
from ket import quant, H, X, phase, adj, measure, control, swap
from functools import partial


def qft(qubits: quant, invert: bool = True):
    if len(qubits) == 1:
        H(qubits)
    else:
        *head, tail = qubits
        H(tail)
        for i, ctrl_qubit in enumerate(reversed(head)):
            with control(ctrl_qubit):
                phase(pi / 2**(i + 1), tail)
        qft(head, invert=False)

    if invert:
        size = len(qubits)
        for i in range(size // 2):
            swap(qubits[i], qubits[size - i - 1])


def phase_estimator(oracle, precision: int) -> int:
    precision -= 1

    ctr = H(quant(precision))
    tgr = X(quant())

    for i, c in enumerate(ctr):
        with control(c):
            oracle(i, tgr)

    adj(qft, ctr)

    return measure(reversed(ctr)).value/2**precision


def oracle(phase_: float, i: int, tgr: quant):
    phase(2*pi*phase_*2**i, tgr)


estimate_pi = partial(phase_estimator, partial(oracle, pi/10))


if __name__ == '__main__':

    print(estimate_pi(15))
