DEVICE=$1
CFG_DIR="$( dirname -- "$0"; )"
EXP_DIR="exp/pfednlp_ft/v5/bert2bert/pt_g3_ft_g1/"

python federatedscope/main.py \
  --cfg $CFG_DIR/config_pretrain.yaml \
  outdir $EXP_DIR/pretrain/ \
  device $DEVICE \

python federatedscope/main.py \
  --cfg $CFG_DIR/config_pfednlp.yaml \
  --cfg_client $CFG_DIR/config_client_pfednlp.yaml \
  federate.load_from $EXP_DIR/pretrain/ckpt/ \
  outdir $EXP_DIR/train/ \
  device $DEVICE \
