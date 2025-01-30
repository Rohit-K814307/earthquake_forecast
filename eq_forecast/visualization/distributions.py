import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import seaborn as sns
import os
import warnings
warnings.filterwarnings("ignore")

def plot_magnitude_distribution(data_dict, output_dir):
    plt.figure(figsize=(8, 6))
    sns.histplot(data_dict["mag"], kde=True, color='blue')
    plt.title("Distribution of Earthquake Magnitudes")
    plt.xlabel("Magnitude")
    plt.ylabel("Frequency")
    plt.legend().set_visible(False)
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    plt.savefig(os.path.join(output_dir, 'magnitude_distribution.png'))
    plt.close()


def plot_depth_distribution(data_dict, output_dir):
    plt.legend('',frameon=False)
    plt.figure(figsize=(8, 6))
    sns.histplot(data_dict["depth"], kde=True, color='green')
    plt.title("Distribution of Earthquake Depths")
    plt.xlabel("Depth (km)")
    plt.ylabel("Frequency")
    plt.legend().set_visible(False)
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    plt.savefig(os.path.join(output_dir, 'depth_distribution.png'))
    plt.close()


def plot_node_plate_distance_distribution(data_dict, output_dir):
    plt.figure(figsize=(8, 6))
    sns.histplot(data_dict["node_plate_dist"], kde=True, color='red')
    plt.title("Distribution of Node-Plate Distances")
    plt.xlabel("Node-Plate Distance (km)")
    plt.ylabel("Frequency")
    plt.legend().set_visible(False)
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    plt.savefig(os.path.join(output_dir, 'node_plate_distance_distribution.png'))
    plt.close()



