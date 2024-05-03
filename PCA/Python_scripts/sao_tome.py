import pandas as pd
import subprocess
import sys
import tempfile
import os

if len(sys.argv) != 4:
    print("Usage: python script.py locus_file vcf_file ids_file")
    exit(1)

df = pd.read_csv(sys.argv[1])
region_column = 'locus'
# If bcftools is not in the system's PATH, replace 'bcftools_path' with the actual path to the bcftools executable
bcftools_path = 'bcftools'
vcf_file = sys.argv[2]
reference_fasta = "Genome/Pfalciparum.genome.fasta"

# Get unique samples from the VCF using bcftools query
'''
bcftools_query_command = [bcftools_path, 'query', '-l', vcf_file]
samples = subprocess.check_output(bcftools_query_command, text=True).splitlines()
'''

# Get samples from the sample id file
sample_file = sys.argv[3]
sample_country_dic = {}
with open (sample_file, 'r') as input_file:
    for line in input_file:
        columns = line.strip().split('\t')
        sample_country_dic[columns[0]]=columns[1]


# Create an empty list to collect the results
result_rows = []

# Iterate over unique genomic regions
for genomic_region in df[region_column].unique():
    if genomic_region[:2] == 'Pf':# Preprocess the region
        processed_region = genomic_region.replace('-', ':', 1).rsplit('-', 1)[0]

    
        print('locus:' ,processed_region)
        # Use samtools faidx to get the subset reference
        subset_reference_command = [
            'samtools',
            'faidx',
            reference_fasta,
            processed_region
        ]
    subset_reference = subprocess.check_output(subset_reference_command, text=True)

    # Write the subset reference to a temporary file
    with tempfile.NamedTemporaryFile(mode='w', delete=False) as temp_fasta_file:
        temp_fasta_file.write(subset_reference)

    # Cut the VCF for the genomic region and save it to a temporary file
    fd, temp_vcf_name = tempfile.mkstemp(suffix='.vcf.gz')
    bcftools_view_command = [
        bcftools_path,
        'view',
        '-r', processed_region,
        '-O', 'z',  # Output in compressed VCF format
        '-o', temp_vcf_name,
        vcf_file
    ]
    subprocess.run(bcftools_view_command)

    # Index the temporary VCF file using tabix
    tabix_command = ['tabix', '-p', 'vcf', temp_vcf_name]
    subprocess.run(tabix_command)

    # Iterate through each sample
    for sample in sample_country_dic:
        print(sample)
        country = sample_country_dic[sample]
        # Construct the bcftools consensus command
        bcftools_consensus_command = [
            bcftools_path,
            'consensus',
            '-s', sample,
            '-f', temp_fasta_file.name,
            temp_vcf_name,
        ]

        # Run the bcftools consensus command and capture the output
        consensus_sequence = subprocess.check_output(bcftools_consensus_command, text=True)
        
        seq_list = consensus_sequence.strip().split('\n')
        sequence=""
        for seq in seq_list:
            if seq[0] != '>':
                sequence += seq
        
        # Append the result to the list
        result_rows.append({'sample': sample, 'country': country, 'locus': genomic_region, 'sequence': sequence })

    # Remove the temporary files
    os.remove(temp_fasta_file.name)
    os.remove(temp_vcf_name)

# Convert the list of rows to a DataFrame
result_df = pd.DataFrame(result_rows, columns=['sample', 'country', 'locus', 'sequence'])

# Create the output file name
# Assuming sample_file contains the full path to the file
base_name = os.path.basename(sample_file)
output_file = os.path.splitext(base_name)[0] + '.tsv'

suffix = 1
while os.path.exists(output_file):
    suffix += 1
    output_file = os.path.splitext(base_name)[0] + '_' + str(suffix) + '.tsv'

# Save the result DataFrame to a tab-delimited file in the specified output directory
result_df.to_csv(output_file, sep='\t', index=False)

print("Output saved to", output_file)

