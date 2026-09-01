#!/bin/bash

set -u

# Temperatures still to sample
TEMPS=(0.01 0.20 0.30 0.50 1.00)

BASE_PARS="inputs/pars.txt"
BASE_SETTINGS="inputs/settings.txt"

mkdir -p sweep_inputs
mkdir -p logs
mkdir -p tests/sweep


# ============================================================
# Create parameter files
# ============================================================

for T in "${TEMPS[@]}"; do

    PARS_FILE="sweep_inputs/pars_T${T}.txt"

    sed \
        -e "s/^T[[:space:]].*/T        ${T}            # sampling temperature/" \
        -e "s/^moves[[:space:]].*/moves     100000        # number of moves/" \
        "$BASE_PARS" > "$PARS_FILE"

done


# ============================================================
# GPU scheduler
#
# Start a simulation only when enough GPU memory is free.
# ============================================================

MIN_FREE_MB=4000
CHECK_INTERVAL=30

running_pids=()


gpu_free_mb() {
    nvidia-smi \
        --query-gpu=memory.free \
        --format=csv,noheader,nounits |
        head -1
}


cleanup_finished() {

    new_pids=()

    for pid in "${running_pids[@]}"; do

        if kill -0 "$pid" 2>/dev/null; then
            new_pids+=("$pid")
        else
            echo "PID $pid finished."
        fi

    done

    running_pids=("${new_pids[@]}")
}


# ============================================================
# Launch simulations
# ============================================================

for T in "${TEMPS[@]}"; do

    echo
    echo "============================================================"
    echo "Preparing T=$T"
    echo "============================================================"

    # Wait until there is enough free GPU memory
    while true; do

        cleanup_finished

        FREE=$(gpu_free_mb)

        echo "GPU free memory: ${FREE} MB"
        echo "Running simulations: ${#running_pids[@]}"

        if [ "$FREE" -ge "$MIN_FREE_MB" ]; then
            break
        fi

        echo "Not enough GPU memory. Waiting ${CHECK_INTERVAL}s..."
        sleep "$CHECK_INTERVAL"

    done


    # --------------------------------------------------------
    # Dedicated directory for this temperature
    # --------------------------------------------------------

    RESULTS_DIR="tests/sweep/T${T}"

    mkdir -p "$RESULTS_DIR"


    # --------------------------------------------------------
    # Create settings file specific to this simulation
    # --------------------------------------------------------

    RUN_SETTINGS="sweep_inputs/settings_T${T}.txt"

    sed \
        -e "s|^results_dir[[:space:]].*|results_dir ${RESULTS_DIR}|" \
        -e "s/^restart[[:space:]].*/restart     0/" \
        "$BASE_SETTINGS" > "$RUN_SETTINGS"


    # --------------------------------------------------------
    # Launch
    # --------------------------------------------------------

    echo "Launching T=$T"
    echo "  results : $RESULTS_DIR"
    echo "  log     : logs/T${T}.log"
    echo "  GPU free: ${FREE} MB"

    nohup python main.py \
        --pars-file "sweep_inputs/pars_T${T}.txt" \
        --settings-file "$RUN_SETTINGS" \
        > "logs/T${T}.log" 2>&1 &

    PID=$!

    running_pids+=("$PID")

    echo "  PID: $PID"

done


# ============================================================
# Wait for all remaining simulations
# ============================================================

echo
echo "All simulations have been launched."
echo "Waiting for remaining simulations..."

while [ "${#running_pids[@]}" -gt 0 ]; do

    cleanup_finished

    if [ "${#running_pids[@]}" -gt 0 ]; then
        sleep "$CHECK_INTERVAL"
    fi

done


echo
echo "============================================================"
echo "All simulations completed."
echo "============================================================"