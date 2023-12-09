#!/bin/bash

# Output file where the results will be saved
output_file="results.txt"

# Clear the output file at the beginning
> $output_file

# Loop over algo values from 0 to 5
for algo in {0..5}
do
    # Loop over bconfig values from 0 to 15
    for bconfig in {4..15}
    do
        echo "Running with --algo=$algo --bconfig=$bconfig --reps=100" | tee -a $output_file
        ./cuhll.bin --algo=$algo --bconfig=$bconfig --reps=100 >> $output_file
        echo "---------------------------------------" >> $output_file
    done
done

echo "All simulations completed."
