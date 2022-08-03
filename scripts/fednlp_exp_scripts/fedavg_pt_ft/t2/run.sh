DEVICE=$1
CFG_DIR="scripts/fednlp_exp_scripts/fedavg_pt_ft/t2/"
EXP_DIR="exp/fedavg_pt_ft/v5/50_50-50_200-5000_3_20000_3"

python federatedscope/main.py \
  --cfg $CFG_DIR/config_pretrain.yaml \
  outdir $EXP_DIR/pretrain/ \
  device $DEVICE \

python federatedscope/main.py \
  --cfg $CFG_DIR/config_fedavg.yaml \
  --cfg_client $CFG_DIR/config_client_fedavg.yaml \
  federate.load_from $EXP_DIR/pretrain/ckpt/ \
  outdir $EXP_DIR/train/ \
  device $DEVICE \

python federatedscope/main.py \
  --cfg $CFG_DIR/config_isolated.yaml \
  --cfg_client $CFG_DIR/config_client_isolated.yaml \
  federate.load_from $EXP_DIR/train/ckpt/ \
  outdir $EXP_DIR/finetune/ \
  device $DEVICE \
