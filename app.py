import json
import os
import pandas as pd
import streamlit as st
import plotly.express as px

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
    'KBW Sparse',
    'KBW Dense',
    'Quantuloop Sparse (f32)',
    'Quantuloop Sparse (f64)',
    'Quantuloop Dense (f32)',
    'Quantuloop Dense (f64)',
    'Quantuloop QuEST'
]

SIMULATORS_SELECTED = [
    'KBW Sparse',
    'KBW Dense',
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
            "Instance (CPU/GPU)",
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
    match st.radio("Comparative between", ["Simulators", "Instances (CPU/GPU)"]):
        case 'Simulators':
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

        case 'Instances (CPU/GPU)':
            plot_selection = 'Instance (CPU/GPU)'

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
    match st.radio("Plot type", ['Line plot', 'Bar plot']):
        case 'Line plot':
            plot_type = px.line
            plot_options = dict(markers=True)

        case 'Bar plot':
            plot_type = px.bar
            plot_options = dict(barmode='group')

    time_log = st.checkbox("Show time in logarithmic scale", value=True)


def filter_data(benchmark):
    df = data.loc[BENCHMARK_SHORT_NAME[benchmark]]
    df = df[df['Instance (CPU/GPU)'].isin(instance)]
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
"""

"""
## Shor's Algorithm
"""

plot('shor')

"""
## Grover's Algorithm
"""

plot('grover')

"""
## Phase Estimator
"""

plot('phase')
