import kagglehub
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

if __name__ == "__main__":
    path = kagglehub.dataset_download("datasnaek/youtube-new")
    print("Path to dataset files:", path)