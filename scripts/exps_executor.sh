#!/bin/bash

programname=$0
function usage {
    echo "usage: $programname [--hyperparams-path=hyperparams_path] [--start=start_id] [--n=exps_number] [--r=repeats_number] [--results-path=results_path]"
    echo "  --hyperparams-path  Path to hyperparameters"
    echo "  --start             Id of first experiment"
    echo "  --n                 Experiments number"
    echo "  --r                 Number of repeats for each experiment"
    echo "  --results-path      Path for saving results"
    exit 1
}

for i in "$@"
do
case $i in
    --hyperparams-path=*)
    HYPERPARAMS_PATH="${i#*=}"
    ;;
    --start=*)
    START_ID="${i#*=}"
    ;;
    --n=*)
    N="${i#*=}"
    ;;
    --r=*)
    REPEATS="${i#*=}"
    ;;
    --path=*)
    RESULTS_PATH="${i#*=}"
    ;;
    *)
    usage        # unknown option
    ;;
esac
done

if [[ -z "$HYPERPARAMS_PATH" || -z "$START_ID" || -z "$N" || -z "$REPEATS" || -z "${RESULTS_PATH}" ]]
then
    usage
fi

while IFS=";" read -r ID LR LANGS BATCH_SIZE MAX_LENGTH EPOCHS EPS WARMUP_STEPS MAX_NORM
do
    echo "$ID"
    if [[ $ID -lt $START_ID || $ID -ge $(($START_ID+$N)) ]]; then
        continue
    fi

    FIRST=$(($ID-$START_ID+1))
    echo "Starting experiment with id=$ID ($FIRST/$N)"

    echo "$line"

    echo "id: $ID"
    echo "lr: $LR"
    echo "langs: $LANGS"
    echo "batch size: $BATCH_SIZE"
    echo "max length: $MAX_LENGTH"
    echo "epochs: $EPOCHS"
    echo "eps: $EPS"
    echo "warmup steps: $WARMUP_STEPS"
    echo "max norm: $MAX_NORM"

    for i in $( eval echo {1..${REPEATS}} ); do
        echo -e "Repeat: $i\n"
        RES=$( python main.py --task "R" --lr ${LR} --langs ${LANGS} --batch-size ${BATCH_SIZE} --max-length ${MAX_LENGTH} --epochs ${EPOCHS} --eps ${EPS} --warmup-steps ${WARMUP_STEPS} --max-norm ${MAX_NORM} | tee /dev/stderr )
    done

    echo -e "\n\n"

done < <(tail -n +2 $HYPERPARAMS_PATH)
