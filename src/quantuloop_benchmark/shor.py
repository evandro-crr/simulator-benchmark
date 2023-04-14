from math import gcd, pi
from ket import quant, H, phase, swap, measure, dump, adj, ctrl
from ket.plugins import pown
from functools import reduce


N_BITS = {4: 15,
          5: 21,
          6: 55,
          7: 119,
          8: 247,
          9: 437,
          10: 899,
          11: 2021,
          12: 4087,
          13: 8051,
          14: 15707,
          15: 32399,
          16: 64507,
          17: 129551,
          18: 262063,
          19: 522713,
          20: 1040399,
          21: 2090867,
          22: 4186067,
          23: 8380961,
          24: 16777207,
          25: 33547183,
          26: 66994189,
          27: 134165873,
          28: 268140589,
          29: 536848891,
          30: 1073610467,
          31: 2147302777,
          32: 4294967071,
          33: 8589026293,
          34: 17178558473,
          35: 34358700317,
          36: 50508067591}

A_MAP = {15: 11,
         21: 5,
         55: 14,
         119: 10,
         247: 227,
         437: 137,
         899: 708,
         2021: 773,
         4087: 1331,
         8051: 4152,
         15707: 11753,
         32399: 536,
         64507: 8580,
         129551: 113373,
         262063: 41886,
         522713: 12224,
         1040399: 147671,
         2090867: 556498,
         4186067: 1427079,
         8380961: 23225}


def qft(qubits: quant, invert: bool = True):
    if len(qubits) == 1:
        H(qubits)
    else:
        head, *tail = qubits
        H(head)
        for i, c in enumerate(reversed(tail)):
            ctrl(c, phase(pi / 2**(i + 1)), head)
        qft(tail, invert=False)

    if invert:
        for i in range(len(qubits) // 2):
            swap(qubits[i], qubits[- i - 1])


def quantum_subroutine(N, a):
    n = N.bit_length()
    reg1 = H(quant(n))
    reg2 = pown(a, reg1, N)
    measure(reg2)
    adj(qft, reg1)
    r = reduce(gcd, dump(reg1).shots)
    return 2**n//r


def shor(N: int) -> int:
    for _ in range(N.bit_length()):
        a = A_MAP[N]
        r = quantum_subroutine(N, a)

        if r % 2 == 0 and pow(a, r//2) != -1 % N:
            p = gcd(a**(r//2)-1, N)
            if p != 1 and p != N and p*N//p == N:
                return p

            q = gcd(a**(r//2)+1, N)
            if q != 1 and q != N and q*N//q == N:
                return q
    raise RuntimeError('failed to factor N')


def shor_n_bits(n: int) -> int:
    return shor(N_BITS[n//2])


if __name__ == '__main__':
    from ket import kbw
    kbw.use_dense()
    print(shor_n_bits(8))
