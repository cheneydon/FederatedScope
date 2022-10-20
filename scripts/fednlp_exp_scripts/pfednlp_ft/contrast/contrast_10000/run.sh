DEVICE=$1
CFG_DIR="$( dirname -- "$0"; )"
PRETRAIN_DIR="exp/pfednlp_pt/v5/bert2bert/200_50/group_5/"
EXP_DIR="exp/pfednlp_ft/v5/bert2bert/contrast/contrast_10000/"

python federatedscope/main.py \
  --cfg $CFG_DIR/config_pfednlp_contrast.yaml \
  --cfg_client $CFG_DIR/config_client_pfednlp_contrast.yaml \
  federate.load_from $PRETRAIN_DIR/pretrain/ckpt/ \
  outdir $EXP_DIR/train/ \
  device $DEVICE \
