import os
import sampler
import sys

if len(sys.argv) != 6 :
    print("Usage: python multysampler.py <country_list_file> <num_samples> <samplesID_file> <samples_years_file>  <output_directory>")
    sys.exit(1)

# Define the filename of the country list file
country_list_file = sys.argv[1]

# Define the number of samples for each country
num_samples_per_country = sys.argv[2]
print(num_samples_per_country, "samples per country")

# Define the samples ID file
samples_id_file = sys.argv[3]

# Define the output directory 
output_directory = sys.argv[5]

# Define smple years
sample_years  = sys.argv[4]

with open(country_list_file, 'r') as file:
    for line in file:
        # Remove leading and trailing whitespace (e.g., newline characters)
        country_name = line.strip()
        # Call the sampler function for each country and save results
        sampler.sampler(samples_id_file, country_name, num_samples_per_country, output_directory, sample_years)
