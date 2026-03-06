import statistics as st

def c_describe(dataset_name, dataset):
    mean = st.mean(dataset)
    median = st.median(dataset)
    mode = st.multimode(dataset)
    variance = st.pvariance(dataset)
    std = st.pstdev(dataset)
    print(f"{dataset_name}:")
    print(f"mean: {mean}")
    print(f"median: {median}")
    print(f"mode: {mode}")
    print(f"variance: {variance}")
    print(f"std: {std}\n")
