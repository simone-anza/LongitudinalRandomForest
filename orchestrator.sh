#!/bin/bash

######################################
## USER CONFIGURABLE INPUTS
######################################

# Task type: REGRESSION or CLASSIFICATION
type="REGRESSION"

# Input files (use absolute or relative paths)
mpa="/path/to/microbiome_table.tsv"
metadata="/path/to/metadata.tsv"

# Variable to split your dataset (e.g., cytokine category)
splitting_var="your_splitting_variable"
splitting_val="0"  # the value of the splitting_var for this run

# Outcome to predict
variable_to_test="age_days"

# Metadata column names
sample_column="sample_id"
individual_column="individual_id"
timepoint_column="timepoint"

# Output directory for results
results_dir="/path/to/results/${splitting_var}"
temp_dir="$results_dir/temp"
mkdir -p "$temp_dir"

######################################
## PATH TO PYTHON TOOLS
######################################

python_exec="python3"
codes_dir="/path/to/code/directory"

regr_script="$codes_dir/regr_for_external_dataset_pred.py"
parser_script="$codes_dir/parse_ML_results.py"
create_folds_script="$codes_dir/create_balanced_folds.py"
avg_pred_script="$codes_dir/average_external_predictions.py"

######################################
## DEBUG INFO
######################################

echo "→ Loading microbiome: $mpa"
echo "→ Loading metadata: $metadata"
echo "→ Splitting by variable: $splitting_var == $splitting_val"
echo "→ Analysis type: $type"
echo "→ Temp results to: $temp_dir"
echo ""

######################################
## STEP 1: SUBSET SPLITTING_VAL METADATA
######################################

split_id="${splitting_var}_${splitting_val}"
split_opposite_val=$((1 - splitting_val))
split_opposite_id="${splitting_var}_${split_opposite_val}"

metadata_split="$temp_dir/metadata_${split_id}.tsv"
awk -v FS='\t' -v OFS='\t' -v col="$splitting_var" -v val="$splitting_val" '
    BEGIN {colnum = -1}
    NR==1 {
        for (i=1; i<=NF; i++) if ($i == col) colnum = i
        if (colnum == -1) {print "Column not found!" > "/dev/stderr"; exit 1}
    }
    NR==1 || $colnum == val {print}
' "$metadata" > "$metadata_split"

n_subjects=$(cut -f1 "$metadata_split" | tail -n +2 | sort | uniq | wc -l)
n_samples=$(wc -l < "$metadata_split")
n_samples=$((n_samples - 1))
echo "📊 Subjects (${split_id}): $n_subjects"
echo "📊 Samples (${split_id}):  $n_samples"
echo ""

######################################
## STEP 2: CREATE 5 BALANCED FOLDS
######################################

echo "🔀 Creating 5 balanced subsets (1 sample per subject, balanced timepoints)..."
$python_exec "$create_folds_script" \
    --metadata "$metadata_split" \
    --sample_column "$sample_column" \
    --individual_column "$individual_column" \
    --timepoint_column "$timepoint_column" \
    --n_folds 5 \
    --output_dir "$temp_dir"

######################################
## STEP 3: TRAIN AND SAVE MODELS
######################################

echo ""
echo "🚀 Launching ML analysis on all 5 folds..."
for i in $(seq 1 5); do
    fold_meta="$temp_dir/fold${i}.tsv"
    fold_output="$results_dir/${type}/fold${i}"
    mkdir -p "$fold_output"

    echo ""
    echo "🔁 Fold $i: $fold_meta → $fold_output"

    $python_exec "$regr_script" \
        -i "$mpa" -m "$fold_meta" -l "$variable_to_test" \
        --inp_metadata_col "$sample_column" \
        -t 4 --zoe --taxa_only_SGBs \
        -o "$fold_output" --verbose

    $python_exec "$parser_script" \
        -i "$fold_output/${variable_to_test}.pkl" \
        -o "$fold_output/${variable_to_test}_${split_id}"
done

echo ""
echo "✅ All folds complete! Ready to move to averaging step."
echo ""

######################################
## STEP 4: PREDICT ON THE OTHER SPLIT
######################################

echo ""
echo "🔮 Predicting on alternate group (${split_opposite_id}) using trained models..."

metadata_opposite="$temp_dir/metadata_${split_opposite_id}.tsv"
awk -v FS='\t' -v OFS='\t' -v col="$splitting_var" -v val="$split_opposite_val" '
    BEGIN {colnum = -1}
    NR==1 {
        for (i=1; i<=NF; i++) if ($i == col) colnum = i
        if (colnum == -1) {print "Column not found!" > "/dev/stderr"; exit 1}
    }
    NR==1 || $colnum == val {print}
' "$metadata" > "$metadata_opposite"

external_pred_dir="$results_dir/EXTERNAL_PREDICTION_${split_id}"
mkdir -p "$external_pred_dir"

for i in $(seq 1 5); do
    model_file="$results_dir/${type}/fold${i}/${variable_to_test}_model_and_norm.pkl"
    output_pred="$external_pred_dir/pred_fold${i}_${split_opposite_id}.tsv"

    echo "🧠 Fold $i: Applying model → $output_pred"
    $python_exec "$regr_script" \
        --model "$model_file" \
        --inp_metadata "$metadata_opposite" \
        --inp_metaphlan "$mpa" \
        --inp_metadata_col "$sample_column" \
        --label "$variable_to_test" \
        --output "$output_pred"
done

echo ""
echo "✅ ALL DONE: ML training and external predictions complete!"
echo "   → Predictions: $external_pred_dir"
echo ""

######################################
## STEP 5: AVERAGE EXTERNAL PREDICTIONS
######################################

echo ""
echo "🧮 Averaging predictions across 5 folds..."

$python_exec "$avg_pred_script" \
    --input_glob "$external_pred_dir/pred_fold*_${split_opposite_id}.tsv" \
    --output_file "$external_pred_dir/avg_predictions_${split_opposite_id}_from_${split_id}.tsv"

echo "🎯 Done! Averaged prediction file:"
echo "     → $external_pred_dir/avg_predictions_${split_opposite_id}_from_${split_id}.tsv"
