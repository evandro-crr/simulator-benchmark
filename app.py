import json
import os
import pandas as pd
import streamlit as st
import plotly.express as px

st.set_page_config(
    page_title="CQSul - Quantuloop Quantum Simulator Benchmark",
    page_icon="https://simulator.quantuloop.com/_static/favicon.ico",
    menu_items={
        "About": "Quantuloop Quantum Simulator Benchmark\n\nCopyright 2023 Quantuloop",
    },
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
    unsafe_allow_html=True,
)

QUANTULOOP_PLOTLY_THEME = dict(
    plot_bgcolor="rgba(0,0,0,0)",
    paper_bgcolor="rgba(0,0,0,0)",
    font_family="Manrope",
    font_color="#0F1F2E",
    legend=dict(yanchor="top", y=1, xanchor="left", x=0),
)

BENCHMARK_SHORT_NAME = {
    "grover": "Grover's Algorithm",
    "shor": "Shor's Algorithm",
    "phase": "Phase Estimator",
}

SIMULATORS = [
    "Quantuloop Sparse (f32)",
    "Quantuloop Sparse (f64)",
    "Quantuloop Dense (f32)",
    "Quantuloop Dense (f64)",
    "Quantuloop QuEST",
]

SIMULATORS_SELECTED = [
    "Quantuloop Sparse (f32)",
    "Quantuloop Dense (f32)",
    "Quantuloop QuEST",
]


@st.cache_data
def load_data():
    df = {
        "grover": {},
        "shor": {},
        "phase": {},
    }
    for data_path in os.listdir("data"):
        benchmark, instance, simulator = (
            os.path.basename(data_path)[:-5].replace("-", " ").split("_")
        )
        if instance not in df[benchmark]:
            df[benchmark][instance] = {}
        with open("data/" + data_path, "r") as file_json:
            data_json = json.load(file_json)
            df[benchmark][instance][simulator] = {}
            df[benchmark][instance][simulator]["n_qubits"] = data_json["n_qubits"]
            df[benchmark][instance][simulator]["time"] = data_json["time"]

    benchmark_index = []
    instances = set()
    row_data = []

    for benchmark in df:
        for instance in df[benchmark]:
            for simulator in df[benchmark][instance]:
                sim_data = df[benchmark][instance][simulator]
                for n_qubits, time in zip(sim_data["n_qubits"], sim_data["time"]):
                    instances.add(instance)
                    benchmark_index.append(BENCHMARK_SHORT_NAME[benchmark])
                    row_data.append((instance, simulator, n_qubits, time))

    df = pd.DataFrame(
        row_data,
        index=[
            benchmark_index,
        ],
        columns=[
            "Number of GPUs",
            "Simulator",
            "N# Qubits",
            "Time (s)",
        ],
    )

    return instances, df, row_data, benchmark_index


@st.cache_data
def load_data_speed_up(data, base_instance, row_data, index):
    data = data.loc[data["Number of GPUs"] == base_instance]
    new_data = []
    benchmark_index = []
    for row, benchmark in zip(row_data, index):
        instance, simulator, n_qubits, time = row
        df = data.loc[benchmark]
        df = df.loc[df["Simulator"] == simulator]
        df = df.loc[df["N# Qubits"] == n_qubits]
        try:
            base_time = float(df["Time (s)"].iloc[0])
            new_data.append((instance, simulator, n_qubits, time, base_time / time))
            benchmark_index.append(benchmark)
        except IndexError:
            pass

    return pd.DataFrame(
        new_data,
        index=[
            benchmark_index,
        ],
        columns=[
            "Number of GPUs",
            "Simulator",
            "N# Qubits",
            "Time (s)",
            "Speed up",
        ],
    )


INSTANCES, DATA, ROW_DATA, INDEX = load_data()

with st.sidebar:
    """
    # Data selection

    ## Filter
    """
    option = st.radio("Comparative between", ["Simulators", "Number of GPUs"])

    plot_y = "Time (s)"

    if option == "Simulators":
        plot_selection = "Simulator"

        plot_title = st.selectbox(
            "Select the Number of GPUs",
            sorted(INSTANCES),
        )

        instance = [plot_title]

        plot_title += " - Lower is better"

        simulator = st.multiselect(
            "Select the Simulators",
            SIMULATORS,
            SIMULATORS_SELECTED,
        )

        if not len(simulator):
            st.warning("Select at least one simulator", icon="⚠️")

        base_instance = None

    elif option == "Number of GPUs":
        plot_selection = "Number of GPUs"

        plot_title = st.selectbox("Select the simulator", SIMULATORS)

        simulator = [plot_title]

        instance = st.multiselect("Select the instances", sorted(INSTANCES), INSTANCES)

        if not len(instance):
            st.warning("Select at least one instance", icon="⚠️")

        base_instance = st.checkbox("Use an instance as base performance")

        if base_instance:
            base_instance = st.selectbox(
                "Select an instance as the base performance", instance
            )
            plot_y = "Speed up"
            plot_title += " - Higher is better"
        else:
            plot_title += " - Lower is better"

    """
    ## Options
    """
    option = st.radio("Plot type", ["Line plot", "Bar plot"])
    if option == "Line plot":
        plot_type = px.line
        plot_options = dict(markers=True)
    elif option == "Bar plot":
        plot_type = px.bar
        plot_options = dict(barmode="group")

    time_log = st.checkbox("Show time in logarithmic scale", value=True)


def filter_data(benchmark):
    if base_instance:
        df = load_data_speed_up(DATA, base_instance, ROW_DATA, INDEX)
    else:
        df = DATA
    df = df.loc[BENCHMARK_SHORT_NAME[benchmark]]
    df = df[df["Number of GPUs"].isin(instance)]
    df = df[df["Simulator"].isin(simulator)]
    return df


def plot(benchmark):
    fig = plot_type(
        filter_data(benchmark),
        title=f"{BENCHMARK_SHORT_NAME[benchmark]} - {plot_title}",
        x="N# Qubits",
        y=plot_y,
        color=plot_selection,
        log_y=time_log,
        **plot_options,
    )

    fig.update_layout(**QUANTULOOP_PLOTLY_THEME)

    st.plotly_chart(fig)


st.image("https://simulator.quantuloop.com/_images/quloop-sim-logo.svg")

"""
# Projeto Computação Quântica na Região Sul do Brasil - https://cqsul.gitlab.io

# Quantum Simulator Benchmark

The benchmark results of the Quantuloop Quantum Simulator Suite for HPC.

**System:**

- 2x  Intel(R) Xeon(R) Gold 5318Y CPU @ 2.10GHz
- 2x NVIDIA Tesla A30 24GB with NVLink
- Memory: 251GiB
- OS: Rocky Linux 8.8
- NVIDIA driver: 535.54.03

For instructions on how to deploy the simulator suite, please visit https://simulator.quantuloop.com. 
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
    plot("shor")

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
    plot("grover")

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
    plot("phase")

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
