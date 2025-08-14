from pathlib import Path

file_path = Path(__file__).resolve()
ROOT_PATH = file_path.parents[1]
DATA_PATH = ROOT_PATH.parents[1] / "repos_data" / "copernicus_cov_reader"
INPUT_PATH = DATA_PATH / "input"
OUTPUT_PATH = DATA_PATH / "output"
