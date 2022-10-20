DEVICE=$1
CFG_DIR="$( dirname -- "$0"; )"
PRETRAIN_DIR="exp/pfednlp_pt/v5/bert2bert/200_50/group_3/"
EXP_DIR="exp/pfednlp_ft/v5/bert2bert/pt_g3_ft_t12/"

python federatedscope/main.py \
  --cfg $CFG_DIR/config_pfednlp.yaml \
  --cfg_client $CFG_DIR/config_client_pfednlp.yaml \
  federate.load_from $PRETRAIN_DIR/pretrain/ckpt/ \
  outdir $EXP_DIR/train/ \
  device $DEVICE \
