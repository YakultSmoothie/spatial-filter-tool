#!/bin/bash
#=========================================================
# File: calc-spatial_filter-multi_var.bash
# Purpose: Run spatial bandpass filter for multiple years, variables and levels
# Author: YakultSmoothie and Claude(CA)
# Created: 2024-12-09
#=========================================================

#-----------------
# Configuration
#-----------------
# Years to process
YEARS=("2020")

# Variables and levels
# VARS=("z" "w" "q" "t" "u" "v")
VARS=("z")
# LEVELS=("1000" "0850" "0700" "0500" "0200")
LEVELS=("0850")

# Fixed parameters
FILTER_DIST="1000"
FILTER_TYPE="lowpass"

# Path settings
INPUT_BASE="./"
OUTPUT_BASE="./OUTPUT/${FILTER_TYPE}/${FILTER_DIST}"

#-----------------
# Main Processing
#-----------------
for YEAR in "${YEARS[@]}"; do
    echo "Processing year: ${YEAR}"

    # Set input/output paths for current year
    INPUT_DATA="${INPUT_BASE}/${YEAR}.pre.nc"
    OUT_DIR="${OUTPUT_BASE}/${YEAR}"

    # Create output directory
    mkdir -p ${OUT_DIR}

    # Generate time sequence using dates.sh
    TIME1="${YEAR}102200"
    TIME2="${YEAR}112000"
    TIMES=(`${mysh}/dates.sh ${TIME1} ${TIME2} 6`)

    # Process each time, variable, and level
    for TIME in "${TIMES[@]}"; do
        for VAR in "${VARS[@]}"; do
            for LEVEL in "${LEVELS[@]}"; do
                echo "Processing: Time=${TIME}, Variable=${VAR}, Level=${LEVEL}"

                python3 ${mypy}/spatial_bandpass_filter.py \
                    -d ${FILTER_DIST} \
                    -ft ${FILTER_TYPE} \
                    -T ${TIME} \
                    -V ${VAR} \
                    -L ${LEVEL} \
                    -i ${INPUT_DATA} \
                    -o ${OUT_DIR}/${TIME}-${VAR}${LEVEL}.bin \
                    -np 

                if [ $? -eq 0 ]; then
                    echo "Successfully processed: ${TIME}-${VAR}${LEVEL}"
                else
                    echo "Error processing: ${TIME}-${VAR}${LEVEL}"
                fi
            done
        done
    done

    echo "Completed processing for year ${YEAR}"
    echo "----------------------------------------"
done

echo "All processing complete!"

#=========================================================
# Usage:
#   bash run-spatial_filter-loop.bash
#
# Requirements:
#   - ERA5 input data files
#
# Current Configuration:
#   - Filter: ${FILTER_TYPE}, Scale: ${FILTER_DIST}km
#
# Time coverage:
#   - Period: 10/22 00Z to 11/20 00Z
#
# Output structure:
#   ${OUTPUT_BASE}/${YEAR}/YYYYMMDDHH-VAR-LEVEL.bin
#
# Example:
#   cd /path/to/working/directory
#   bash run-spatial_filter-loop.bash
#
# Note:
#   - Modify VARS and LEVELS arrays as needed
#   - Check INPUT_BASE and OUTPUT_BASE paths before running
#=========================================================
