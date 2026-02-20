import statistics as st

def analyze(dataset_name, data):
    mean = st.mean(data)
    median = st.median(data)
    mode = st.multimode(data)
    variance = st.pvariance(data)
    std = st.pstdev(data)
    print(f"{dataset_name}:")
    print(f"mean: {mean}")
    print(f"median: {median}")
    print(f"mode: {mode}")
    print(f"variance: {variance}")
    print(f"std: {std}\n")
