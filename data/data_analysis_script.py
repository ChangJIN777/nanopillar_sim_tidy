import pandas as pd
import sys
import os 
import csv

# function definition 
def read_csv_without_header(file_path):
    with open(file_path, mode='r') as file:
        print()
        # # Create a CSV reader object
        # csv_reader = csv.reader(file)
    
        # # Read and print each row
        # for row in csv_reader:
        #     print(row)


# # Example usage:
# # Specify the path to the folder you want to add
# folder_path = "/Users/changjin/Desktop/Research/GitHub/nanopillar_sim_tidy/"

# # Ensure the folder path is absolute
# folder_path = os.path.abspath(folder_path)

# # Add the folder path to the system path
# if folder_path not in sys.path:
#     sys.path.append(folder_path)
    
# # Get the current working directory
# current_directory = os.getcwd()

# # Print the current working directory
# print("Current working directory:", current_directory)
    
file_path = "nanopillar_diameter_200nm.csv"  # Replace with your CSV file path
read_csv_without_header(file_path)