#!/bin/bash

set -u

# Fixed trajectory length:
# tau = dt * isteps = 1.0

DTS=(0.002 0.004 0.005 0.008 0.010 0.0125 0.020 0.025)

BASE_PARS="inputs/pars.txt"
BASE_SETTINGS="inputs/settings.txt"

mkdir -p sweep_fixed_tau_inputs
mkdir -p logs_fixed_tau
mkdir -p tests/sweep_fixed_tau

TAU=1.0
MOVES=5000

# Maximum number of simultaneous simulations
MAX_RUNNING=4

# Minimum free GPU memory required before launching a new simulation
MIN_FREE_MB=4000

CHECK_INTERVAL=30

running_pids=()


# ============================================================
# Create parameter files
# ============================================================

for DT in "${DTS[@]}"; do

    ISTEPS=$(python -c "print(round($TAU / $DT))")

    # Check that dt * isteps = tau
    python -c "assert abs($DT * $ISTEPS - $TAU) < 1e-10"

    PARS_FILE="sweep_fixed_tau_inputs/pars_dt${DT}.txt"

    sed \
        -e "s/^dt[[:space:]]*.*/dt        ${DT}            # integration time-step/" \
        -e "s/^isteps[[:space:]]*.*/isteps    ${ISTEPS}            # integration steps/" \
        -e "s/^moves[[:space:]]*.*/moves     ${MOVES}        # number of moves/" \
        "$BASE_PARS" > "$PARS_FILE"

    echo "dt=${DT}, isteps=${ISTEPS}, tau=$(python -c "print($DT * $ISTEPS)")"

done


# ============================================================
# GPU scheduler
# ============================================================

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

for DT in "${DTS[@]}"; do

    echo
    echo "============================================================"
    echo "Preparing dt=$DT"
    echo "============================================================"


    # --------------------------------------------------------
    # Wait until:
    #   1. fewer than MAX_RUNNING simulations are active
    #   2. enough GPU memory is free
    # --------------------------------------------------------

    while true; do

        cleanup_finished

        FREE=$(gpu_free_mb)

        echo "GPU free memory: ${FREE} MB"
        echo "Running simulations: ${#running_pids[@]} / ${MAX_RUNNING}"

        if [ "${#running_pids[@]}" -lt "$MAX_RUNNING" ] && \
           [ "$FREE" -ge "$MIN_FREE_MB" ]; then
            break
        fi

        echo "Waiting ${CHECK_INTERVAL}s..."
        sleep "$CHECK_INTERVAL"

    done


    # --------------------------------------------------------
    # Dedicated results directory
    # --------------------------------------------------------

    RESULTS_DIR="tests/sweep_fixed_tau/dt${DT}"

    mkdir -p "$RESULTS_DIR"


    # --------------------------------------------------------
    # Dedicated settings file
    # --------------------------------------------------------

    RUN_SETTINGS="sweep_fixed_tau_inputs/settings_dt${DT}.txt"

    sed \
        -e "s|^results_dir[[:space:]].*|results_dir ${RESULTS_DIR}|" \
        "$BASE_SETTINGS" > "$RUN_SETTINGS"


    ISTEPS=$(grep '^isteps' "$sweep_fixed_tau_inputs/pars_dt${DT}.txt")


    echo "Launching dt=$DT"
    echo "  isteps:  $ISTEPS"
    echo "  tau:     $TAU"
    echo "  results: $RESULTS_DIR"
    echo "  log:     logs_fixed_tau/dt${DT}.log"
    echo "  GPU free: ${FREE} MB"


    nohup python main.py \
        --pars-file "sweep_fixed_tau_inputs/pars_dt${DT}.txt" \
        --settings-file "$RUN_SETTINGS" \
        > "logs_fixed_tau/dt${DT}.log" 2>&1 &

    PID=$!

    running_pids+=("$PID")

    echo "  PID: $PID"

done


# ============================================================
# Wait for all simulations
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
echo "All fixed-tau simulations completed."
echo "============================================================"