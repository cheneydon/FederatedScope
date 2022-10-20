DEVICE=$1
CFG_DIR="$( dirname -- "$0"; )"
EXP_DIR="exp/spfl/100_200/"

python federatedscope/main.py \
  --cfg $CFG_DIR/config_pfednlp.yaml \
  --cfg_client $CFG_DIR/config_client_pfednlp.yaml \
  outdir $EXP_DIR/train/ \
  device $DEVICE \
