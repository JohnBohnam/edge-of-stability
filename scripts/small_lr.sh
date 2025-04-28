#!/usr/bin/env bash
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
source "$SCRIPT_DIR/environ.sh"

{ time bash "$SCRIPT_DIR/flow.sh" > "$OUTPUT/flow_output.log" 2>&1; } 2> "$OUTPUT/flow_time.log" &
{ time bash "$SCRIPT_DIR/gd.sh" > "$OUTPUT/gd_output.log" 2>&1; } 2> "$OUTPUT/gd_time.log" &
{ time bash "$SCRIPT_DIR/adam.sh" > "$OUTPUT/adam_output.log" 2>&1; } 2> "$OUTPUT/adam_time.log" &

wait