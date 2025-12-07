# import kagglehub

# datasets ={
#     # 'random_forest' : 'elikplim/car-evaluation-data-set',
#     'knn' : "gkalpolukcu/knn-algorithm-dataset", 
#     'svm' :"vinod00725/svm-classification",
#     'elliptic_envelope':"brjapon/gearbox-fault-diagnosis",


# }

# # Download latest version
# for key, dataset in datasets.items():
#     print(f"Downloading dataset for {key}...")
#     # name its folder with the key name
#     path = kagglehub.dataset_download(dataset, path=key) 
    
#     # print the downloaded path 
#     print(f"Dataset for {key} downloaded to: {path}")

import kagglehub

# Download latest version
path = kagglehub.dataset_download("brjapon/gearbox-fault-diagnosis")

print("Path to dataset files:", path)