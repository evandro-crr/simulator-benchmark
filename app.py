import json
import os
import pandas as pd
import streamlit as st
import plotly.express as px

st.set_page_config(
    page_title="Quantuloop Quantum Simulator Benchmark",
    page_icon="https://simulator.quantuloop.com/_static/favicon.ico",
    menu_items={
        'About': 'Quantuloop Quantum Simulator Benchmark\n\nCopyright 2023 Quantuloop',
    }
)

st.markdown(
    """
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Manrope&display=swap');
        html, body, [class*="css"]  {
            font-family: 'Manrope', sans-serif;
        }
    </style>
    """,
    unsafe_allow_html=True
)

quantuloop_plotly_theme = dict(
    plot_bgcolor='rgba(0,0,0,0)',
    paper_bgcolor='rgba(0,0,0,0)',
    font_family='Manrope',
    font_color='#0F1F2E',
    legend=dict(
        yanchor="top",
        y=1,
        xanchor="left",
        x=0
    ),
)

BENCHMARK_SHORT_NAME = {
    "grover": "Grover's Algorithm",
    "shor":  "Shor's Algorithm",
    "phase": "Phase Estimator",
}

SIMULATORS = [
    'Quantuloop Sparse (f32)',
    'Quantuloop Sparse (f64)',
    'Quantuloop Dense (f32)',
    'Quantuloop Dense (f64)',
    'Quantuloop QuEST'
]

SIMULATORS_SELECTED = [
    'Quantuloop Sparse (f32)',
    'Quantuloop Dense (f32)',
    'Quantuloop QuEST'
]


@st.cache_data
def load_data():
    data = {
        'grover': {},
        'shor': {},
        'phase': {},
    }
    for data_path in os.listdir("data"):
        benchmark, instance, simulator = os.path.basename(
            data_path)[:-5].replace("-", " ").split("_")
        if instance not in data[benchmark]:
            data[benchmark][instance] = {}
        with open("data/"+data_path, 'r') as file_json:
            data_json = json.load(file_json)
            data[benchmark][instance][simulator] = {}
            data[benchmark][instance][simulator]["n_qubits"] = data_json['n_qubits']
            data[benchmark][instance][simulator]["time"] = data_json['time']

    benchmark_index = []
    instances = set()
    data_frame = []

    for benchmark in data:
        for instance in data[benchmark]:
            for simulator in data[benchmark][instance]:
                sim_data = data[benchmark][instance][simulator]
                for n_qubits, time in zip(sim_data["n_qubits"], sim_data["time"]):
                    instances.add(instance)
                    benchmark_index.append(BENCHMARK_SHORT_NAME[benchmark])
                    data_frame.append((instance, simulator, n_qubits, time))

    data = pd.DataFrame(
        data_frame,
        index=[
            benchmark_index,
        ],
        columns=[
            "AWS EC2 Instances",
            "Simulator",
            "N# Qubits",
            "Time (s)",
        ]
    )

    return instances, data


instances, data = load_data()

with st.sidebar:
    """
    # Data selection

    ## Filter
    """
    option = st.radio("Comparative between", [
                      "Simulators", "AWS EC2 Instances"])
    if option == 'Simulators':
        plot_selection = 'Simulator'

        plot_title = st.selectbox(
            "Select the Instance",
            sorted(instances)
        )

        instance = [plot_title]

        simulator = st.multiselect(
            "Select the Simulators",
            SIMULATORS,
            SIMULATORS_SELECTED,
        )

        if not len(simulator):
            st.warning('Select at least one simulator', icon="⚠️")

    elif option == 'AWS EC2 Instances':
        plot_selection = 'AWS EC2 Instances'

        plot_title = st.selectbox(
            "Select the simulator",
            SIMULATORS
        )

        simulator = [plot_title]

        instance = st.multiselect(
            "Select the instances",
            sorted(instances),
            instances
        )

        if not len(instance):
            st.warning('Select at least one instance', icon="⚠️")

    """
    ## Options
    """
    option = st.radio("Plot type", ['Line plot', 'Bar plot'])
    if option == 'Line plot':
        plot_type = px.line
        plot_options = dict(markers=True)
    elif option == 'Bar plot':
        plot_type = px.bar
        plot_options = dict(barmode='group')

    time_log = st.checkbox("Show time in logarithmic scale", value=True)


def filter_data(benchmark):
    df = data.loc[BENCHMARK_SHORT_NAME[benchmark]]
    df = df[df['AWS EC2 Instances'].isin(instance)]
    df = df[df['Simulator'].isin(simulator)]
    return df


def plot(benchmark):
    fig = plot_type(
        filter_data(benchmark),
        title=f'{BENCHMARK_SHORT_NAME[benchmark]} ({plot_title})',
        x='N# Qubits',
        y='Time (s)',
        color=plot_selection,
        log_y=time_log,
        **plot_options,
    )

    fig.update_layout(**quantuloop_plotly_theme)

    st.plotly_chart(fig)


st.image("https://simulator.quantuloop.com/_images/quloop-sim-logo.svg")

"""
# Quantum Simulator Benchmark

The benchmark results of the Quantuloop Quantum Simulator Suite for HPC on AWS.
For instructions on how to deploy the simulator suite on the AWS Cloud, please visit https://simulator.quantuloop.com. 
"""

"""
## Shor's Algorithm

[Shor's factorization algorithm](https://en.wikipedia.org/wiki/Shor%27s_algorithm) 
is an example where the Quantuloop Sparse simulator outperforms other simulation models.
To factorize an n-bit number, at least $2n$ qubits are required, and each step (quantum gate)
on the Quantuloop QuEST simulator takes $O(2^{2n})$. In contrast, each step on the
Quantuloop Sparse simulator takes $O(\log(2^n)2^N)$, and the measure on the second quantum
register further reduces the time complexity by collapsing the quantum state of the second
register, decreasing the superposition of the first one due to entanglement.
"""

plot_tab, code_tab = st.tabs(["Performance Plot", "Ket Code"])

with plot_tab:
    plot('shor')

with code_tab:
    """
    ```py
    def quantum_subroutine(N, a):
        n = N.bit_length()
        reg1 = H(quant(n))
        reg2 = pown(a, reg1, N)
        measure(reg2)
        adj(qft, reg1)
        r = reduce(gcd, dump(reg1).shots)
        return 2**n//r

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
    ```
    """

"""
## Grover's Algorithm

The Quantuloop Dense and Quantuloop QuEST simulators are both effective options for
simulating [Grover's search algorithm](https://en.wikipedia.org/wiki/Grover%27s_algorithm).
In benchmark tests, the execution time of the Quantuloop Dense simulator grows more slowly
than that of the Quantuloop Sparse simulator. Grover's algorithm is a quantum search
algorithm that can provide a quadratic speedup over classical search algorithms for
unstructured search problems.
"""

plot_tab, code_tab = st.tabs(["Performance Plot", "Ket Code"])

with plot_tab:
    plot('grover')

with code_tab:
    """
    ```py
    def grover(n: int, oracle) -> int:
        s = H(quant(n))
        steps = int((pi/4)*sqrt(2**n))
        for _ in range(steps):
            oracle(s)
            with around(H, s):
                phase_on(0, s)
        return measure(s).value
    ```
    """

"""
## Phase Estimator

The [Phase Estimation Algorithm](https://en.wikipedia.org/wiki/Quantum_phase_estimation_algorithm)
is a common algorithm used in quantum computing. When simulating this algorithm,
the Quantuloop Dense and Quantuloop QuEST simulators are the optimal choices. 
In comparison to the Quantuloop Sparse simulator, both Quantuloop QuEST and
Quantuloop Dense offer a more efficient execution time.
"""

plot_tab, code_tab = st.tabs(["Performance Plot", "Ket Code"])

with plot_tab:
    plot('phase')

with code_tab:
    """
    ```py
    def phase_estimator(oracle, precision: int) -> int:
        precision -= 1
        ctr = H(quant(precision))
        tgr = X(quant())
        for i, c in enumerate(ctr):
            with control(c):
                oracle(i, tgr)
        adj(qft, ctr)
        return measure(reversed(ctr)).value/2**precision

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
    ```
    """
