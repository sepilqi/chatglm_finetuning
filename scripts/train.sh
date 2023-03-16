DATA_DIR=${1:-/data/glm}
CHECKPOINT=${2:-checkpoint/chatglm-6b}

CHECKPOINT_PATH=${DATA_DIR}/${CHECKPOINT}
DATE_STR=$(date +"%Y.%m.%d")
LOG_DIR=${DATA_DIR}/logs/${DATE_STR}
VERSION=$(date +"%H%M%S")
EXPERIMENT_NAME=finetune-6b-scratch-${DATE_STR}-${VERSION}

if [ -d $LOG_DIR ]; then
  echo "Log directory exists"
else
  echo "Log directory does not exist; making directory" $LOG_DIR
  mkdir -p $LOG_DIR
fi

run_cmd="python train.py 2>&1 | tee ${LOG_DIR}/log.${EXPERIMENT_NAME}.log"

echo ${run_cmd} 2>&1 | tee ${LOG_DIR}/run.${EXPERIMENT_NAME}.sh
eval ${run_cmd}
