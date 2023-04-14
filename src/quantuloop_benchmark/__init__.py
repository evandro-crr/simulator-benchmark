import json
from shutil import copy
from os import path, makedirs
from ket import quantum_exec_time
from .grover import grover_w3
from .shor import shor_n_bits
from .phase import estimate_pi


BENCHMARK = {
    "Grover's Algorithm": grover_w3,
    "Shor's Algorithm": shor_n_bits,
    "Phase Estimator": estimate_pi,
}

BENCHMARK_SHORT_NAME = {
    "grover": "Grover's Algorithm",
    "shor":  "Shor's Algorithm",
    "phase": "Phase Estimator",
}

BENCHMARK_INPUTS = {
    "grover": list(range(2, 31)),
    "shor":  list(range(8, 74, 2)),
    "phase": list(range(10, 45)),
}


def benchmark_setup(data_json, name, inputs, timeout):
    try:
        with open(data_json, 'r') as file:
            data = json.load(file)
            data['inputs'] = inputs
            data['timeout'] = timeout
    except FileNotFoundError:
        data = {
            'benchmark': name,
            'inputs': inputs,
            'next': 0,
            'timeout': timeout,
            'n_qubits': [],
            'time': []
        }

    with open(data_json, 'w') as file:
        json.dump(data, file, indent=4)


def benchmark_run(data_json, setup):
    with open(data_json, 'r') as file:
        data = json.load(file)
    setup()

    while data['next'] < len(data['inputs']):
        if len(data['time']) and data['time'][-1] >= data['timeout']:
            break

        n_qubits = data['inputs'][data['next']]

        BENCHMARK[data['benchmark']](n_qubits)
        data['n_qubits'].append(n_qubits)
        exec_time = quantum_exec_time()
        data['time'].append(exec_time)

        data['next'] += 1
        copy(data_json, ".backup/")
        with open(data_json, 'w') as file:
            json.dump(data, file, indent=4)


def execute(instance_name: str, simulator_name: str, setup, benchmark, timeout):
    makedirs("data", exist_ok=True)
    makedirs(".backup", exist_ok=True)

    data_paths = []
    for name in benchmark:
        data_paths.append(
            f'data/{name}_{instance_name.replace(" ", "-")}_{simulator_name.replace(" ", "-")}.json')
        benchmark_setup(
            data_paths[-1],
            BENCHMARK_SHORT_NAME[name],
            BENCHMARK_INPUTS[name],
            timeout,
        )

    for data_path in data_paths:
        benchmark_run(data_path, setup)
