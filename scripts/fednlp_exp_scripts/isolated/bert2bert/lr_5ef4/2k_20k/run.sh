DEVICE=$1
CFG_DIR="$( dirname -- "$0"; )"
EXP_DIR="exp/isolated/v5/bert2bert/lr_5ef4/2k_20k/im_ag/"

python federatedscope/main.py \
  --cfg $CFG_DIR/config_isolated.yaml \
  --cfg_client $CFG_DIR/config_client_isolated.yaml \
  outdir $EXP_DIR/train/ \
  device $DEVICE \
  seed 1 \

python federatedscope/main.py \
  --cfg $CFG_DIR/config_isolated.yaml \
  --cfg_client $CFG_DIR/config_client_isolated.yaml \
  outdir $EXP_DIR/train/ \
  device $DEVICE \
  seed 12 \

python federatedscope/main.py \
  --cfg $CFG_DIR/config_isolated.yaml \
  --cfg_client $CFG_DIR/config_client_isolated.yaml \
  outdir $EXP_DIR/train/ \
  device $DEVICE \
  seed 123 \

python federatedscope/main.py \
  --cfg $CFG_DIR/config_isolated.yaml \
  --cfg_client $CFG_DIR/config_client_isolated.yaml \
  outdir $EXP_DIR/train/ \
  device $DEVICE \
  seed 1234 \

python federatedscope/main.py \
  --cfg $CFG_DIR/config_isolated.yaml \
  --cfg_client $CFG_DIR/config_client_isolated.yaml \
  outdir $EXP_DIR/train/ \
  device $DEVICE \
  seed 12345 \

python federatedscope/main.py \
  --cfg $CFG_DIR/config_isolated.yaml \
  --cfg_client $CFG_DIR/config_client_isolated.yaml \
  outdir $EXP_DIR/train/ \
  device $DEVICE \
  seed 10 \

python federatedscope/main.py \
  --cfg $CFG_DIR/config_isolated.yaml \
  --cfg_client $CFG_DIR/config_client_isolated.yaml \
  outdir $EXP_DIR/train/ \
  device $DEVICE \
  seed 100 \

python federatedscope/main.py \
  --cfg $CFG_DIR/config_isolated.yaml \
  --cfg_client $CFG_DIR/config_client_isolated.yaml \
  outdir $EXP_DIR/train/ \
  device $DEVICE \
  seed 1000 \

python federatedscope/main.py \
  --cfg $CFG_DIR/config_isolated.yaml \
  --cfg_client $CFG_DIR/config_client_isolated.yaml \
  outdir $EXP_DIR/train/ \
  device $DEVICE \
  seed 10000 \

python federatedscope/main.py \
  --cfg $CFG_DIR/config_isolated.yaml \
  --cfg_client $CFG_DIR/config_client_isolated.yaml \
  outdir $EXP_DIR/train/ \
  device $DEVICE \
  seed 100000 \
