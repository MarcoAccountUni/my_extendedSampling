#!/bin/bash

set -u

# ============================================================
# Sweep over trajectory length tau
#
# Fixed:
#   dt = 0.002
#   moves = 5000
#
# Varied:
#   tau
#
# isteps = tau / dt
# ============================================================

DT=0.002

TAUS=(0.1 0.25 0.5 1.0 2.0 4.0)

BASE_PARS="inputs/pars.txt"
BASE_SETTINGS="inputs/settings.txt"

mkdir -p sweep_tau_inputs
mkdir -p logs_tau
mkdir -p tests/sweep_tau

MOVES=5000

# Maximum number of simultaneous simulations
MAX_RUNNING=4

# Minimum free GPU memory required before launching
# a new simulation
MIN_FREE_MB=4000

CHECK_INTERVAL=30

running_pids=()


# ============================================================
# Create parameter files
# ============================================================

for TAU in "${TAUS[@]}"; do

    ISTEPS=$(python -c "print(round($TAU / $DT))")

    # Check that dt * isteps = tau
    python -c "assert abs($DT * $ISTEPS - $TAU) < 1e-10"

    PARS_FILE="sweep_tau_inputs/pars_tau${TAU}.txt"

    sed \
        -e "s/^dt[[:space:]]*.*/dt        ${DT}            # integration time-step/" \
        -e "s/^isteps[[:space:]]*.*/isteps    ${ISTEPS}            # integration steps/" \
        -e "s/^moves[[:space:]]*.*/moves     ${MOVES}        # number of moves/" \
        "$BASE_PARS" > "$PARS_FILE"

    echo "tau=${TAU}, dt=${DT}, isteps=${ISTEPS}, tau_check=$(python -c "print($DT * $ISTEPS)")"

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

for TAU in "${TAUS[@]}"; do

    echo
    echo "============================================================"
    echo "Preparing tau=$TAU"
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

    RESULTS_DIR="tests/sweep_tau/tau${TAU}"

    mkdir -p "$RESULTS_DIR"


    # --------------------------------------------------------
    # Dedicated settings file
    # --------------------------------------------------------

    RUN_SETTINGS="sweep_tau_inputs/settings_tau${TAU}.txt"

    sed \
        -e "s|^results_dir[[:space:]].*|results_dir ${RESULTS_DIR}|" \
        "$BASE_SETTINGS" > "$RUN_SETTINGS"


    # --------------------------------------------------------
    # Read isteps for display
    # --------------------------------------------------------

    ISTEPS=$(grep '^isteps' \
        "sweep_tau_inputs/pars_tau${TAU}.txt")


    echo
    echo "Launching tau=$TAU"
    echo "  dt:       $DT"
    echo "  isteps:   $ISTEPS"
    echo "  tau:      $TAU"
    echo "  moves:    $MOVES"
    echo "  results:  $RESULTS_DIR"
    echo "  log:      logs_tau/tau${TAU}.log"
    echo "  GPU free: ${FREE} MB"


    # --------------------------------------------------------
    # Launch
    # --------------------------------------------------------

    nohup python main.py \
        --pars-file "sweep_tau_inputs/pars_tau${TAU}.txt" \
        --settings-file "$RUN_SETTINGS" \
        > "logs_tau/tau${TAU}.log" 2>&1 &

    PID=$!

    running_pids+=("$PID")

    echo "  PID: $PID"
    echo "  Running now: ${#running_pids[@]} / $MAX_RUNNING"

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
echo "All tau simulations completed."
echo "============================================================"