"""
Author: Cooter McGrew
Date: 2025-02-13
Description: 
    This program encodes CSV data into a structured gene representation for use in genetic algorithms. 
    Each row is treated as a 'gene' that can be uniquely addressed, manipulated, and used for genetic operations 
    (mutation, crossover, selection). 

    Features:
    - Encodes CSV data into a structured gene dictionary
    - Supports spatial/temporal encoding (rows maintain ordering)
    - Allows individual genes to be addressed for modification
    - Outputs a structured codified representation for later use in GA

Usage:
    ```
    codified_genes = codify_genes("data.csv", key_column="ID")
    gene = access_gene(codified_genes, gene_id="A23")
    ```
"""

import pandas as pd
import hashlib
import json

class GeneCodifier:
    """
    A class to codify CSV data into structured gene representations.
    """

    def __init__(self, key_column=None, spatial_coding=True, logical_coding=False):
        """
        Initialize the gene codifier.
        
        Parameters:
        - key_column (str): The column used to uniquely identify each gene. If None, a hash ID is generated.
        - spatial_coding (bool): Whether to maintain spatial order of rows.
        - logical_coding (bool): Future implementation: Logical clustering of similar genes.
        """
        self.key_column = key_column
        self.spatial_coding = spatial_coding
        self.logical_coding = logical_coding

    def hash_gene(self, row):
        """
        Generate a unique hash for a gene when no explicit key is provided.
        Uses SHA256 hashing for stability.

        Parameters:
        - row (pd.Series): A row from the dataframe.

        Returns:
        - str: A unique hash-based identifier.
        """
        return hashlib.sha256(str(row.to_dict()).encode()).hexdigest()[:10]  # Shortened for readability

    def codify_genes(self, file_path):
        """
        Codifies genes from a CSV file.

        Parameters:
        - file_path (str): Path to the CSV file.

        Returns:
        - dict: A structured representation of genes.
        """
        df = pd.read_csv(file_path)

        # Check if a key column exists
        if self.key_column and self.key_column in df.columns:
            df['Gene_ID'] = df[self.key_column].astype(str)
        else:
            df['Gene_ID'] = df.apply(self.hash_gene, axis=1)  # Generate unique ID if none provided

        # Convert rows into gene format
        genes = {}
        for idx, row in df.iterrows():
            gene_data = row.drop('Gene_ID').to_dict()  # Exclude identifier from gene data
            genes[row['Gene_ID']] = {
                "data": gene_data,  # Gene attributes
                "index": idx if self.spatial_coding else None,  # Retain index for spatial analysis
            }

        return genes

    def save_codified_genes(self, genes, output_file="codified_genes.json"):
        """
        Saves the codified genes to a JSON file.

        Parameters:
        - genes (dict): The codified gene structure.
        - output_file (str): Filename for saving the gene data.
        """
        with open(output_file, 'w') as f:
            json.dump(genes, f, indent=4)
        print(f"Codified genes saved to {output_file}")

    def load_codified_genes(self, input_file="codified_genes.json"):
        """
        Loads codified genes from a JSON file.

        Parameters:
        - input_file (str): Filename to load gene data from.

        Returns:
        - dict: The gene structure.
        """
        with open(input_file, 'r') as f:
            genes = json.load(f)
        return genes

    def access_gene(self, genes, gene_id):
        """
        Retrieves a specific gene by its ID.

        Parameters:
        - genes (dict): The structured gene dictionary.
        - gene_id (str): The unique identifier of the gene.

        Returns:
        - dict: The gene data.
        """
        return genes.get(gene_id, None)  # Return None if gene does not exist


# Example Usage (for testing)
if __name__ == "__main__":
    codifier = GeneCodifier(key_column="ID")  # Use "ID" column if available, else auto-generate
    codified_genes = codifier.codify_genes("example_data.csv")
    
    # Save the codified genes
    codifier.save_codified_genes(codified_genes, "genes_output.json")

    # Load and access a sample gene
    loaded_genes = codifier.load_codified_genes("genes_output.json")
    sample_gene_id = list(loaded_genes.keys())[0]  # Pick the first available gene ID
    print(f"Sample Gene [{sample_gene_id}]:", codifier.access_gene(loaded_genes, sample_gene_id))
