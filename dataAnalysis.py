import paths
from utils import *
import matplotlib.pyplot as plt
import pandas as pd
import os

def create_plot():
      """
      Creates a plot showing the distribution of ClassIds in Test, Train, and Meta datasets.
      """
      # Accessing ClassId column inside CSV files
      class_counts_Test = read_test_csv['ClassId'].value_counts()
      class_counts_Train = read_train_csv['ClassId'].value_counts()
      class_counts_Meta = read_meta_csv['ClassId'].value_counts()

      class_counts_Meta = class_counts_Meta.sort_index()
      class_counts_Train = class_counts_Train.sort_index()
      class_counts_Test = class_counts_Test.sort_index()

      # Create a figure with three subplots
      fig, axs = plt.subplots(1, 3, figsize=(15, 5))

      # Plotting for 'Meta' dataset
      axs[0].bar(class_counts_Meta.index.astype(int), class_counts_Meta.values, color='blue')
      axs[0].set_title('Meta Dataset')
      axs[0].set_xlabel('Class Labels')
      axs[0].set_ylabel('Number of Samples')

      # Plotting for 'Train' dataset
      axs[1].bar(class_counts_Train.index.astype(int), class_counts_Train.values)
      axs[1].set_title('Train Dataset')
      axs[1].set_xlabel('Class Labels')
      axs[1].set_ylabel('Number of Samples')

      # Plotting for 'Test' dataset
      axs[2].bar(class_counts_Test.index.astype(int), class_counts_Test.values)
      axs[2].set_title('Test Dataset')
      axs[2].set_xlabel('Class Labels')
      axs[2].set_ylabel('Number of Samples')

      for ax in axs:
          ax.set_xticks(range(0, max(class_counts_Meta.index.max(), 
                                    class_counts_Train.index.max(), 
                                    class_counts_Test.index.max()) + 1))

      # Adjust layout
      plt.tight_layout()

      return fig

if __name__ == "__main__":


  # Extracting first 20 classes
  new_train = process_data(paths.train_folder)
  meta_csv = process_data(paths.meta_path)
  test_csv = process_data(paths.test_path)
  train_csv = process_data(paths.train_path)

  # reading csv files using pandas library
  read_meta_csv = pd.read_csv(meta_csv)
  read_test_csv = pd.read_csv(test_csv)
  read_train_csv = pd.read_csv(train_csv)

  

  save_figures('figures', create_plot, 'class_distribution')
