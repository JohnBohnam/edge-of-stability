SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
(time (sh "$SCRIPT_DIR/ngd1.sh" > /dev/null) 2> "$SCRIPT_DIR/../bench/ngd1_time.txt") &
(time (sh "$SCRIPT_DIR/ngd2.sh" > /dev/null) 2> "$SCRIPT_DIR/../bench/ngd2_time.txt") &
(time (sh "$SCRIPT_DIR/ngd3.sh" > /dev/null) 2> "$SCRIPT_DIR/../bench/ngd3_time.txt") &

