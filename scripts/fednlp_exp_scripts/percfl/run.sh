DEVICE=$1
CFG_DIR="$( dirname -- "$0"; )"
EXP_DIR="exp/percfl/100_200_2.5_7.5_20/"

python federatedscope/main.py \
  --cfg $CFG_DIR/config_pfednlp.yaml \
  --cfg_client $CFG_DIR/config_client_pfednlp.yaml \
  outdir $EXP_DIR/train/ \
  device $DEVICE \
