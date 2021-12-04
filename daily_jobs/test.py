import os

dir_path = os.getcwd()
file_path = dir_path.replace('daily_jobs', '') + 'data/raw_data.csv'
print(file_path)
