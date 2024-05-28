import matplotlib.pyplot as plt
import csv
import os
from utils import *

    
if __name__ == '__main__':
    results = os.path.join("src", "logs", "mobilenetdice.csv")
    save_figures('figures', plot_training_results, 'mobilenetdice_results', results)
    
