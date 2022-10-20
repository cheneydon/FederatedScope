DEVICE=$1
CFG_DIR="$( dirname -- "$0"; )"
EXP_DIR="exp/pfednlp_pt/v5/bert2bert/test/"

#python federatedscope/main.py \
#  --cfg $CFG_DIR/config_pretrain.yaml \
#  outdir $EXP_DIR/pretrain/ \
#  device $DEVICE \
#  data.debug True \

python federatedscope/main.py \
  --cfg $CFG_DIR/config_isolated.yaml \
  --cfg_client $CFG_DIR/config_client_isolated.yaml \
  outdir $EXP_DIR/finetune/ \
  device $DEVICE \
  data.debug True \
