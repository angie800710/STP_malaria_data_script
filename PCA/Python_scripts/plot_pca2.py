import sys
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from itertools import combinations
from itertools import combinations_with_replacement

#This script produces several plots of pca file, samples are colored acording to the grouping file provided and the number of PCs is specified as argument

def plot_pca_subplots(df_pca, num_pcs, group_variable):
    unique_groups = df_pca[group_variable].unique()
    group_palette = sns.color_palette('husl', n_colors=len(unique_groups))

    # Create a grid of subplots
    fig, axes = plt.subplots(num_pcs, num_pcs, figsize=(22, 22), sharex='all', sharey='all')

    # Iterate through all combinations of PCs
    for (pc1, pc2) in combinations(range(1, num_pcs + 1), 2):
        row, col = pc1 - 1, pc2 - 1

        # Original subplot
        ax = axes[row, col]
        sns.scatterplot(x=f'PC{pc2}', y=f'PC{pc1}', data=df_pca, hue=group_variable, palette=custom_colors, s=5, ax=ax)
        ax.set_xlabel(f'PC{pc2}')
        ax.set_ylabel(f'PC{pc1}')
        ax.get_legend().remove()  # Remove legends from subplots

        # Mirrored subplot
        ax_mirror = axes[col, row]
        sns.scatterplot(x=f'PC{pc1}', y=f'PC{pc2}', data=df_pca, hue=group_variable, palette=custom_colors, s=5, ax=ax_mirror)
        ax_mirror.set_xlabel(f'PC{pc1}')
        ax_mirror.set_ylabel(f'PC{pc2}')
        ax_mirror.get_legend().remove()  # Remove legends from subplots

      # Remove spines and ticks for diagonal plots
    for i in range(num_pcs):
        for j in range(num_pcs):
            if i == j:
                axes[i, j].spines['top'].set_visible(False)
                axes[i, j].spines['right'].set_visible(False)
                axes[i, j].spines['bottom'].set_visible(False)
                axes[i, j].spines['left'].set_visible(False)
                axes[i, j].tick_params(left=False, right=False, top=False, bottom=False)

    # Set common labels for x and y axes
    for i in range(num_pcs):
        axes[i, 0].set_ylabel(f'PC{i + 1}')
        axes[num_pcs - 1, i].set_xlabel(f'PC{i + 1}')

    # Add a common legend outside the subplots
    handles, labels = axes[1, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper right', bbox_to_anchor=(.98, .98))
    
    # Adjust layout with more space
    plt.tight_layout(pad=5.0, h_pad=2.0, w_pad=2.0)

    plt.tight_layout()
    plt.subplots_adjust(top=.95, right=0.85)  # Adjust top spacing for the title and legend position
    plt.show()

# Read data from file
pca_data = pd.read_csv(sys.argv[1], sep=',')
pca_country = pd.read_csv(sys.argv[2], sep='\t')
group= sys.argv[3]


pca_country['country'] = pca_country['country'].map(lambda x: 'West Africa' if x in ['Ghana', 'Nigeria', 'Mauritania', 'Guinea', "Côte d'Ivoire", "Mali", 'Burkina Faso', 'Senegal', 'Gambia', 'Benin'] else
                                                     ('East Africa' if x in ['Ethiopia', 'Tanzania', 'Kenya', 'Mozambique', 'Madagascar', 'Uganda', 'Sudan', 'Malawi' ] else
                                                     ('Central Africa' if x in ['Gabon', 'Cameroon', 'Democratic Republic of the Congo'] else
                                                     ('Sao Tome' if x == 'Sao_Tome' else None))))



# Create a scatter plot
country = pca_country[['sample', 'country']]
df_pca = pd.merge(pca_data, country, on="sample")

# Get unique clusters
unique_clusters = df_pca['country'].unique()

# Set a color palette with a unique color for each cluster
cluster_palette = sns.color_palette('Set3', n_colors=len(unique_clusters))

# Define your own color palette
custom_colors = ['#FFD700', 'teal', 'purple', 'red']

# Number of PCs to be plotted
num_pcs_to_plot = int(sys.argv[4])

# Call the function to plot subplots
plot_pca_subplots(df_pca, num_pcs_to_plot, group)

