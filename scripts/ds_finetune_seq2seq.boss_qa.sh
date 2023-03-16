DATA_DIR=/data/boss-gpt
DATA_ROOT=${DATA_DIR}/input/boss-qa/v2023.03.02
# CHECKPOINT_PATH=${DATA_DIR}/checkpoint/boss-gpt-10-ch
CHECKPOINT_PATH=${DATA_DIR}/checkpoint/boss-gpt-10-2023.03.08/finetune-arti-GLM-10B-chinese-boss-qa-2023.03.08-100816-2.8.1
DATE_STR=$(date +"%Y.%m.%d")
VERSION=$(date +"%H%M%S")
SAVE_PATH=${DATA_DIR}/checkpoint/boss-gpt-10-${DATE_STR}

source $1    # Model
source $2    # Task

NUM_WORKERS=1
NUM_GPUS_PER_WORKER=8
HOSTS=${DATA_DIR}/hosts
MP_SIZE=1
MASTER_PORT=$(shuf -n 1 -i 10000-65535)

OPTIONS_NCCL="NCCL_DEBUG=info NCCL_IB_DISABLE=0 NCCL_NET_GDR_LEVEL=2 NCCL_IB_GID_INDEX=3"
DISTRIBUTED_ARGS="${OPTIONS_NCCL} deepspeed --hostfile ${HOSTS} --master_port ${MASTER_PORT} --num_nodes ${NUM_WORKERS} --num_gpus ${NUM_GPUS_PER_WORKER}"

EXPERIMENT_NAME=finetune-arti-${EXPERIMENT_NAME}-${DATE_STR}-${VERSION}-${NUM_WORKERS}.${NUM_GPUS_PER_WORKER}.${MP_SIZE}
LOG_DIR=${DATA_DIR}/logs/${DATE_STR}
if [ -d $LOG_DIR ]; then
  echo "Log directory exists"
else
  echo "Log directory does not exist; making directory" $LOG_DIR
  mkdir -p $LOG_DIR
fi
run_cmd="${DISTRIBUTED_ARGS} finetune_glm.py \
       --deepspeed \
       --deepspeed_config config_tasks/config_blocklm_10B_boss_qa.json \
       --finetune \
       --experiment-name ${EXPERIMENT_NAME} \
       --task ${TASK_NAME} \
       --data-dir ${DATA_PATH} \
       --save ${SAVE_PATH} \
       --checkpoint-activations \
       --num-workers 1 \
       --no-load-lr-scheduler \
       $MODEL_ARGS \
       $TRAIN_ARGS \
       $COMMON_ARGS \
       $TASK_ARGS \
       --fp16 \
       --model-parallel-size ${MP_SIZE} \
       --overwrite \
       --summary-dir ${LOG_DIR} \
       2>&1 | tee ${LOG_DIR}/log.deepspeed.10b.${EXPERIMENT_NAME}.log"

echo ${run_cmd} 2>&1 | tee ${LOG_DIR}/run.deepspeed.10b.${EXPERIMENT_NAME}.sh
eval ${run_cmd}
