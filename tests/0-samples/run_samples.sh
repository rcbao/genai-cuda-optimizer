#!/bin/bash

# Define the number of runs
num_runs=20

# Create arrays to store times for each sample
declare -a times_matrix_mul
declare -a times_vector_add
declare -a times_nbody

# Function to calculate statistical metrics
function calc_stats {
    local times=("$@")
    local length=${#times[@]}
    local total=0
    local average median stddev min max p95

    # Calculate total
    for time in "${times[@]}"; do
        # Ensure the time is a valid number, removing extra characters and potential spaces
        clean_time=$(echo "$time" | sed 's/[^0-9.]*//g' | tr -d '[:space:]')
        total=$(echo "$total + $clean_time" | bc -l)
    done

    # Average
    average=$(echo "scale=5; $total / $length" | bc -l)

    # Sort times to calculate median and percentiles
    IFS=$'\n' sorted_times=($(sort -n <<<"${times[*]}"))
    unset IFS

    # Median
    if (( $length % 2 == 0 )); then
        mid_index=$(($length / 2))
        median=$(echo "scale=5; (${sorted_times[$mid_index]} + ${sorted_times[$mid_index - 1]}) / 2" | bc -l)
    else
        mid_index=$(($length / 2))
        median=${sorted_times[$mid_index]}
    fi

    # Min and Max
    min=${sorted_times[0]}
    max=${sorted_times[-1]}

    # 95th percentile
    p95_index=$(echo "$length * 0.95 / 1" | bc)
    p95=${sorted_times[$p95_index]}

    # Standard deviation
    sum_sq=0
    for time in "${times[@]}"; do
        clean_time=$(echo "$time" | sed 's/[^0-9.]*//g' | tr -d '[:space:]')
        sum_sq=$(echo "$sum_sq + ($clean_time - $average)^2" | bc -l)
    done
    stddev=$(echo "scale=5; sqrt($sum_sq / $length)" | bc -l)

    # Print results
    # Print results in table format
    echo -n "Metric             | Average   | Median    | Std Dev   | Min   | Max       | 95th Percentile "
    echo -e "\nValues (ms)      | $average  | $median   | $stddev   | $min  | $max  | $p95"
    echo ""

}

# Run samples and collect data
for i in $(seq 1 $num_runs); do
    # Capture the output of the bash script
    readarray -t output < <(/bin/bash ../run_samples_base.sh)

    # Store results in respective arrays
    times_matrix_mul+=("$(echo ${output[0]} | grep -oP 'Elapsed time: \K[\d.]+')")
    times_vector_add+=("$(echo ${output[1]} | grep -oP 'Elapsed time: \K[\d.]+')")
    times_nbody+=("$(echo ${output[2]} | grep -oP 'Elapsed time: \K[\d.]+')")
done

# Calculate statistics for each program
echo "### Matrix Multiplication Stats ###"
calc_stats "${times_matrix_mul[@]}"

echo "### Vector Addition Stats ###"
calc_stats "${times_vector_add[@]}"

echo "### N-body Simulation Stats ###"
calc_stats "${times_nbody[@]}"
