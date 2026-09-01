#!/bin/bash

set -u

# ============================================================
# Sweep over entropy multiplier lambda_S
# Logarithmic range: 0.001 -> 10000
# 500 Monte Carlo moves for each simulation
# ============================================================

LAMBDA_VALUES=(
    0.001
    0.00316
    0.01
    0.0316
    0.1
    0.316
    1
    3.162
    10
    31.62
    100
    316.2
    1000
    3162
    10000
)

BASE_PARS="inputs/pars.txt"
BASE_SETTINGS="inputs/settings.txt"

mkdir -p sweep_lambda_inputs
mkdir -p logs_lambda
mkdir -p tests/sweep_lambda

MOVES=500

# Maximum number of simultaneous simulations
MAX_RUNNING=4

# Minimum free GPU memory required before launching
# a new simulation
MIN_FREE_MB=4000

CHECK_INTERVAL=60

running_pids=()
running_lambdas=()


# ============================================================
# Create parameter files
# ============================================================

for LAMBDA in "${LAMBDA_VALUES[@]}"; do

    PARS_FILE="sweep_lambda_inputs/pars_lambda${LAMBDA}.txt"

    sed \
        -e "s/^lambda_S[[:space:]]*.*/lambda_S  ${LAMBDA}        # entropy multiplier/" \
        -e "s/^moves[[:space:]]*.*/moves     ${MOVES}        # number of moves/" \
        "$BASE_PARS" > "$PARS_FILE"

    echo "lambda_S=${LAMBDA}, moves=${MOVES}"

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
    new_lambdas=()

    for i in "${!running_pids[@]}"; do

        pid="${running_pids[$i]}"
        lambda="${running_lambdas[$i]}"

        if kill -0 "$pid" 2>/dev/null; then

            new_pids+=("$pid")
            new_lambdas+=("$lambda")

        else

            wait "$pid"
            status=$?

            if [ "$status" -eq 0 ]; then
                echo "PID $pid (lambda_S=$lambda) finished successfully."
            else
                echo "WARNING: PID $pid (lambda_S=$lambda) exited with status $status."
            fi

        fi

    done

    running_pids=("${new_pids[@]}")
    running_lambdas=("${new_lambdas[@]}")

}


# ============================================================
# Launch simulations
# ============================================================

for LAMBDA in "${LAMBDA_VALUES[@]}"; do

    echo
    echo "============================================================"
    echo "Preparing lambda_S=$LAMBDA"
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

    RESULTS_DIR="tests/sweep_lambda/lambda${LAMBDA}"

    mkdir -p "$RESULTS_DIR"


    # --------------------------------------------------------
    # Dedicated settings file
    # --------------------------------------------------------

    RUN_SETTINGS="sweep_lambda_inputs/settings_lambda${LAMBDA}.txt"

    sed \
        -e "s|^results_dir[[:space:]].*|results_dir ${RESULTS_DIR}|" \
        "$BASE_SETTINGS" > "$RUN_SETTINGS"


    # --------------------------------------------------------
    # Launch
    # --------------------------------------------------------

    echo
    echo "Launching lambda_S=$LAMBDA"
    echo "  moves:    $MOVES"
    echo "  results:  $RESULTS_DIR"
    echo "  log:      logs_lambda/lambda${LAMBDA}.log"
    echo "  GPU free: ${FREE} MB"


    nohup python main.py \
        --pars-file "sweep_lambda_inputs/pars_lambda${LAMBDA}.txt" \
        --settings-file "$RUN_SETTINGS" \
        > "logs_lambda/lambda${LAMBDA}.log" 2>&1 &

    PID=$!

    running_pids+=("$PID")
    running_lambdas+=("$LAMBDA")

    echo "  PID: $PID"
    echo "  Running now: ${#running_pids[@]} / ${MAX_RUNNING}"

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
echo "All lambda_S simulations completed."
echo "============================================================"