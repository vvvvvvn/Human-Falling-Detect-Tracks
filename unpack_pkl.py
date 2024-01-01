import pickle

with open("Data-UR/UR_new_new-set(labelXscrw).pkl", "rb") as f:
    data = pickle.load(f)
    print(data)
