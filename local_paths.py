import pathlib
import os
import sys

# settings
dir_separator = os.sep
current_pythonpath = sys.path
sys.path.append(dir_separator.join(current_pythonpath[0].split(dir_separator)[:-1]))


# repository
base_dir = pathlib.Path(__file__).parents[0]
data_dir = base_dir.joinpath('data')
output_dir = base_dir.joinpath('outputs')

# create repositories
if not data_dir.exists():
	os.mkdir(data_dir)

if not output_dir.exists():
	os.mkdir(output_dir)
