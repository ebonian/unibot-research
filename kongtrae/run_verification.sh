#!/bin/bash

# Configuration
SWAP_FILE="../research/simulation_12/training_data/swaps_20250504_to_20260212_eth_usdt_0p3.csv"
SAMPLE_FILE="test_swaps_sample.csv"
LINES=600000

echo "🧪 Running Kongtrae Verification"
echo "=================================================="

if [ ! -f "$SWAP_FILE" ]; then
    echo "❌ Swap file not found at: $SWAP_FILE"
    exit 1
fi

echo "📊 Extracting last $LINES swaps from training data..."
# Get header from original file
head -n 1 "$SWAP_FILE" > "$SAMPLE_FILE"
# Append last N lines
tail -n "$LINES" "$SWAP_FILE" >> "$SAMPLE_FILE"

echo "✅ Created sample file: $SAMPLE_FILE"
echo ""

echo "🤖 Running inference on sample..."
echo "Command: python inference.py --model dqn --swap-csv $SAMPLE_FILE --has-position --current-width 1"
echo ""

../.venv/bin/python inference.py \
    --model dqn \
    --swap-csv "$SAMPLE_FILE" \
    --has-position \
    --current-width 1 \
    --hours-since-rebalance 3

echo ""
echo "✅ Verification complete!"
rm "$SAMPLE_FILE"
