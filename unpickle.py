import pickle 

with open('results/filenames.pkl', 'rb') as f:
    x = pickle.load(f)
    print(x)