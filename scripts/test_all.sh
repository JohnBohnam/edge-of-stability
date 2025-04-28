SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
time (sh ./adam.sh > /dev/null) 2> "$SCRIPT_DIR/../bench/adam_time.txt"
time (sh ./gd.sh > /dev/null) 2> "$SCRIPT_DIR/../bench/gd_time.txt"
time (sh ./flow.sh > /dev/null) 2> "$SCRIPT_DIR/../bench/flow_time.txt"

