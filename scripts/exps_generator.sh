#!/bin/bash

# hiperparameters
LR=("1e-5" "1e-4")
LANGS=("(ru,fa)")
BATCH_SIZE=("32")
MAX_LENGTH=("256")
EPOCHS=("4")
EPS=("1e-8")
WARMUP_STEPS=("0")
MAX_NORM=("1.0" "0.75")


programname=$0
function usage {
    echo "usage: $programname [--id=first_id] [--results-path=results_path]"
    echo "  --id                Number of first experiment"
    echo "  --results-path      Path for saving experiments"
    exit 1
}

for i in "$@"
do
case $i in
    --id=*)
    CURR_ID="${i#*=}"
    ;;
    --path=*)
    RESULTS_PATH="${i#*=}"
    ;;
    *)
    usage        # unknown option
    ;;
esac
done

if [[ -z "$CURR_ID" || -z "$RESULTS_PATH" ]]
then
    usage
fi



all_exps=$(( \
    ${#LR[@]} * \
    ${#LANGS[@]} * \
    ${#BATCH_SIZE[@]} * \
    ${#MAX_LENGTH[@]} * \
    ${#EPOCHS[@]} * \
    ${#EPS[@]} * \
    ${#WARMUP_STEPS[@]} * \
    ${#MAX_NORM[@]} \
))

echo -e "Generate ${all_exps} experiments..."

if [ $CURR_ID -eq 0 ]
then
echo "id;lr;langs;batch_size;max_length;epochs;eps;warmup_steps;max_norm" >> ${RESULTS_PATH}
fi

for lr in "${LR[@]}"; do
    for langs in "${LANGS[@]}"; do
        for batch_size in "${BATCH_SIZE[@]}"; do
            for max_length in "${MAX_LENGTH[@]}"; do
                for epochs in "${EPOCHS[@]}"; do
                    for eps in "${EPS[@]}"; do
                        for warmup_steps in "${WARMUP_STEPS[@]}"; do
                            for max_norm in "${MAX_NORM[@]}"; do
                                echo "${CURR_ID};${lr};${langs};${batch_size};${max_length};${epochs};${eps};${warmup_steps};${max_norm}" >> ${RESULTS_PATH}
                                ((CURR_ID++))
                            done
                        done
                    done
                done
            done
        done
    done
done
