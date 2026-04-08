#!/bin/bash
# Evaluate multiple LinkNet3D checkpoints on the Airbus validation set.
# Run from the OpenPCDet/tools directory.

TOOLS_DIR="/mnt/backup/dassault/ysf/airbus_hackathon/OpenPCDet/tools"
CKPT_DIR="/mnt/backup/dassault/ysf/airbus_hackathon/OpenPCDet/output/airbus_models/linknet3d_airbus/default/ckpt"
CFG_FILE="cfgs/airbus_models/linknet3d_airbus.yaml"
RESULTS_DIR="/mnt/backup/dassault/ysf/airbus_hackathon/eval_results"

mkdir -p "$RESULTS_DIR"

cd "$TOOLS_DIR"

for EPOCH in 55 60 65 70 75 80; do
    CKPT="${CKPT_DIR}/checkpoint_epoch_${EPOCH}.pth"
    if [ ! -f "$CKPT" ]; then
        echo "[SKIP] Epoch $EPOCH: checkpoint not found"
        continue
    fi
    echo ""
    echo "============================================================"
    echo "Evaluating Epoch $EPOCH"
    echo "============================================================"
    
    python test.py \
        --cfg_file "$CFG_FILE" \
        --batch_size 4 \
        --workers 4 \
        --ckpt "$CKPT" \
        --eval_tag "epoch_${EPOCH}_eval" \
        2>&1 | tee "${RESULTS_DIR}/eval_epoch_${EPOCH}.log"
    
    echo "[DONE] Epoch $EPOCH evaluation saved to ${RESULTS_DIR}/eval_epoch_${EPOCH}.log"
done

echo ""
echo "============================================================"
echo "ALL EVALUATIONS COMPLETE"
echo "============================================================"
echo "Results saved in: $RESULTS_DIR"
