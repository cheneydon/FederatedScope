DEVICE=$1
CFG_DIR="$( dirname -- "$0"; )"
PRETRAIN_DIR="exp/pfednlp/"
EXP_DIR="exp/pfednlp/"

python federatedscope/main.py \
  --cfg $CFG_DIR/config_pfednlp.yaml \
  --client_cfg $CFG_DIR/config_client_pfednlp.yaml \
  federate.hfl_load_from $PRETRAIN_DIR/pretrain/ckpt/ \
  outdir $EXP_DIR/train/ \
  device $DEVICE \
