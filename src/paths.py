import os

dataset_path = os.path.join("data")

meta_path = os.path.join("data", "Meta.csv")
test_path = os.path.join("data", "Test.csv")
train_path = os.path.join("data", "Train.csv")


train_folder = os.path.join("data", "Train")
test_folder = os.path.join("data", "Test")
meta_folder = os.path.join("data", "Meta")

# new folders
train_csv = os.path.join("data", "new_dataset", "Train.csv")
meta_csv = os.path.join("data", "new_dataset", "Meta.csv")
test_csv = os.path.join("data", "new_dataset", "Test.csv")
train_new = os.path.join("data", "new_dataset", "Train")

data_directory = "dataDIR"

mymodel = os.path.join("checkpoints", "resnetscratch.pt")
