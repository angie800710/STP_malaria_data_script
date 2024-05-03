import sys
import random
import os


def sampler(input_txt, country_name, num_samples, output_directory, years_file):
    # Open the tab-delimited text file
    with open(input_txt, 'r') as f:
        # Split lines by tabs and create a list for each line
        lines = [line.strip().split('\t') for line in f]

    # Check if the file has at least 15 columns (0-based index)
    if len(lines[0]) < 15:
        print("The input file does not have at least 15 columns.")
        sys.exit(1)

    # Create a dictionary mapping sample IDs to country names for samples that passed QC
    data = {}

    # Create a list to store the selected years
    selected_years = []
    with open (years_file, 'r') as input_file:
        for line in input_file:
            selected_years.append(line.rstrip())
    print("Selected years:", selected_years)
    # Assuming the first column (index 0) contains sample IDs, the third column (index 2) contains country names,
    # and the fifteenth column (index 14) contains QC information
    for line in lines[1:]:
        sample_id = line[0]
        country = line[2] 
        qc_flag = line[13] 
        year = line[8]
        if qc_flag == "True" and year in selected_years:
            data[sample_id] = country, year

    # Create a directory to store the results
    os.makedirs(output_directory, exist_ok=True)  # Create the directory if it doesn't exist
    
    
    # Make a name tag for the output file using 'output_directory'
    output_name = os.path.join(output_directory, "{}.txt".format(country_name))

    # Check if file exists
    if os.path.exists(output_name):
        os.remove(output_name) # Remove file if it exists 
        print(f"File {output_name} was removed")
    
    # Select samples from the specified country
    country_samples = {k: v for k, v in data.items() if v[0] == country_name}

    # Check that the country exists in the data file
    if len(country_samples) == 0:
        print("No samples available for", country_name)
        sys.exit(0)

    # Check if "all" is passed as num_samples
    if num_samples == "all":
        samples = country_samples
    else:
        print(num_samples)
        # Check if num_samples is a valid positive integer
        try:
            num_samples = int(num_samples)
            if num_samples <= 0 or num_samples > len(country_samples):
                print(f"Invalid number of samples requested for {country_name}.")
                return
        except ValueError:
            print(num_samples, "Invalid value for num_samples.")
            sys.exit(1)

        # Sample the specified number of samples
        sampled_ids = random.sample(list(country_samples.keys()), num_samples)
        samples = {key: country_samples[key] for key in sampled_ids}
    # Write the samples to a text file
    with open(output_name, "w") as file:
    #    file.write("Sample\tCountry\tYear\n")
        for key, values in samples.items():
            # Separates the sample's name using a newline character
            file.write(f'{key}\t{values[0]}\t{values[1]}\n')


# Uncomment to use the script on its own 
'''
if __name__ == "__main__":
    if len(sys.argv) != 5:
        print("Usage: python script.py input_txt country_name num_samples output_directory")
        sys.exit(1)

    input_txt = sys.argv[1]
    country_name = sys.argv[2]
    num_samples = sys.argv[3]
    output_directory = sys.argv[4]

    sampler(input_txt, country_name, num_samples, output_directory)
'''
