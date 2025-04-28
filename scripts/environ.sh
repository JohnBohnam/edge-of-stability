
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
BASE_PATH="${SCRIPT_DIR}/../"
export DATASETS="${BASE_PATH}datasets"
export RESULTS="${BASE_PATH}results"
export PYTHON="${BASE_PATH}../venv/bin/python3"
export OUTPUT="${BASE_PATH}output"
#export PYTHON="/usr/bin/python3"
