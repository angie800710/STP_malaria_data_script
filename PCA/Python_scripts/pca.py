import sys
import numpy as np
from sklearn.decomposition import PCA
import pandas as pd
from Bio import AlignIO
from Bio.Seq import Seq
import sys
import subprocess
from Bio.Align import MultipleSeqAlignment 

# One-Hot Encoding base map

import numpy as np

def one_hot_encode(sequence):
    nucleotides = {'A': 0, 'T': 1, 'C': 2, 'G': 3}
    encoding = np.zeros((len(sequence), len(nucleotides) + 1))  # Additional index for other symbols
    for i, nucleotide in enumerate(sequence):
        upper_nucleotide = nucleotide.upper()
        if upper_nucleotide in nucleotides:
            encoding[i, nucleotides[upper_nucleotide]] = 1
        else:
            encoding[i, -1] = 1  # Encoding other symbols as 4
    return encoding.flatten()


def read_sequence_table(file_path):
    df = pd.read_table(file_path, sep='\t')
    return df

# Read the sequences and store them into a pandas dataframe
data= read_sequence_table(sys.argv[1])
print("data read from file",data)

# Write sequences to a temporary FASTA file
with open("sequences.fasta", "w") as fasta_file:
    for i, row in data.iterrows():
        fasta_file.write(f">{row['sample']}\n{row['sequence']}\n")

# Using multseq malign
print('Performing multyple sequence alignment')
multseq_cmd = f"multseq malign --in sequences.fasta --threads 12 --package muscle --out alignment.fasta --verbose"
subprocess.run(multseq_cmd, shell=True, check=True)

def calculate_homogeneity(alignment):
    homogeneity = []
    gap_percentage = []
    alignment_length = alignment.get_alignment_length()
    for i in range(alignment_length):
        column = alignment[:, i]
        gap_count = column.count('-')
        gap_percentage.append(gap_count / len(column))
        counts = {x: column.count(x) for x in set(column)}
        max_count = max(counts.values())
        homogeneity.append(max_count / len(column))
    print("homogeneity:", homogeneity)
    print("gap_percentage:", gap_percentage)
    return homogeneity, gap_percentage


def filter_alignment(alignment):
    homogeneity, gap_percentage = calculate_homogeneity(alignment)
    filtered_alignment = MultipleSeqAlignment([])
    for i in range(len(alignment)):
        filtered_seq = ''
        for j in range(len(homogeneity)):
            if homogeneity[j] <= 0.8 and gap_percentage[j] <=0.5:
                filtered_seq += alignment[i][j]
        filtered_alignment.add_sequence(alignment[i].id, filtered_seq)
    return filtered_alignment

output_file = "filtered_alignment.fasta"

# Read alignment
alignment = AlignIO.read("alignment.fasta", "fasta")

# Filter alignment
filtered_alignment = filter_alignment(alignment)

# Write filtered alignment to output file
AlignIO.write(filtered_alignment, output_file, "fasta")

print("Filtered alignment saved to", output_file)

filtered_alignment = AlignIO.read("filtered_alignment.fasta", "fasta")

# Convert alignment to a list of sequences
sequences = [str(record.seq) for record in filtered_alignment]
print(sequences)

# One-Hot Encoding the sequences
print('One-hot encodding the sequences')
X = np.array([one_hot_encode(seq) for seq in sequences])
print(len(X))
print(X)
for x in X:
    print(x)
    print(len(x))


# PCA
print('Performing PCA')
pca = PCA(n_components=10)
X_pca = pca.fit_transform(X)

# Create a DataFrame with the PCA-transformed data
df_pca = pd.DataFrame(X_pca, columns=[f'PC{i+1}' for i in range(X_pca.shape[1])])
df_pca['sample'] = data['sample']

# Add explained variance ratio to DataFrame
explained_variance_ratio = pca.explained_variance_ratio_

# Save to CSV
df_variance=pd.DataFrame(explained_variance_ratio)
df_variance.to_csv('variance.csv', index=False)

# Save to CSV
df_pca.to_csv('pca_transformed_data.csv', index=False)

