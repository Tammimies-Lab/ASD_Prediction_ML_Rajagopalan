import pandas as pd
import pickle
import os

# File paths
raw_dna_file = '23andme_raw_data.txt'  # Path to your raw 23andMe DNA file
output_file = 'v8_data_comp_all.pkl'   # Output file for the processed dataset

# Step 1: Load the raw 23andMe DNA data
def load_23andme_data(file_path):
    # Read the raw DNA data, skipping comments that start with '#'
    df = pd.read_csv(file_path, sep='\t', comment='#', names=['rsid', 'chromosome', 'position', 'genotype'])
    return df

# Step 2: Filter for relevant SNPs (replace with your list of autism-related SNPs)
# Replace this with a real list of autism-related SNPs if you have one
relevant_snps = ['rs123456', 'rs234567', 'rs345678']  # Example SNPs, replace with actual SNPs

def filter_relevant_snps(dna_df, snp_list):
    # Filter the DNA data for the relevant SNPs
    return dna_df[dna_df['rsid'].isin(snp_list)]

# Step 3: Encode the genotypes numerically
def encode_genotypes(dna_df):
    # Map genotype to numerical values (you can adjust this depending on your needs)
    genotype_map = {
        'AA': 0,
        'AG': 1, 'GA': 1,
        'GG': 2,
        'TT': 0,
        'TC': 1, 'CT': 1,
        'CC': 2,
        # Add more mappings as necessary
    }
    
    # Apply the mapping
    dna_df['genotype_encoded'] = dna_df['genotype'].map(genotype_map)
    
    # Drop any rows with genotypes that aren't in the map
    dna_df = dna_df.dropna(subset=['genotype_encoded'])
    
    return dna_df

# Step 4: Create the feature matrix (X) and labels (y)
def create_feature_matrix(dna_df):
    # For simplicity, X will be the encoded genotypes, and y will be a dummy label (to be replaced with actual labels)
    X = dna_df[['genotype_encoded']].values  # Features (SNPs)
    y = [0] * len(X)  # Dummy labels (replace with actual labels if you have them)
    
    # For demonstration, we're using the rsid as the feature name
    features = dna_df['rsid'].tolist()
    
    return X, y, features

# Step 5: Save the processed data using pickle
def save_as_pickle(X, y, features, output_file):
    with open(output_file, 'wb') as f:
        pickle.dump((X, y, features), f)
    print(f"Processed data saved to {output_file}")

# Main script
def main():
    # Load raw DNA data
    if not os.path.exists(raw_dna_file):
        print(f"File not found: {raw_dna_file}")
        return
    
    dna_data = load_23andme_data(raw_dna_file)
    
    # Filter for relevant SNPs
    filtered_data = filter_relevant_snps(dna_data, relevant_snps)
    
    # Encode the genotypes
    encoded_data = encode_genotypes(filtered_data)
    
    # Create the feature matrix and dummy labels
    X, y, features = create_feature_matrix(encoded_data)
    
    # Save as pickle
    save_as_pickle(X, y, features, output_file)

if __name__ == "__main__":
    main()
