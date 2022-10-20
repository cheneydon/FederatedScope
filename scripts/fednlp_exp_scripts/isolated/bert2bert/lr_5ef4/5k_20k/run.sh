DEVICE=$1
CFG_DIR="$( dirname -- "$0"; )"
EXP_DIR="exp/isolated/v5/bert2bert/raw/lr_5ef4/"

python federatedscope/main.py \
  --cfg $CFG_DIR/config_isolated.yaml \
  --cfg_client $CFG_DIR/config_client_isolated.yaml \
  outdir $EXP_DIR/train/ \
  device $DEVICE \
