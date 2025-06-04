#!/usr/bin/env python3

#################################""
####Random Forest Regression for Longitudinal Compositional Data

#This script trains or applies a Random Forest model using microbiome/metaphlan data
#and subject metadata. It supports external prediction and saves models + normalizers.

#Author: Simone Anza

# THIS IS JUST A DEMOSTRATIVE SCRIPT TO WORK WITH THE ORCHESTRATOR AND IS A REVISED VERSION
# OF THE WORK OF ('Francesco Asnicar (f.asnicar@unitn.it), 'Leonard Dubois (),
# Andrew Maltez Thomas (andrewmaltez.thomas@unitn.it), '  
# IT IS INTENDED AS AN EXAMPLE OF RF APPROACH TO BE MATCHED WITH THE ORCHESTRATOR

import argparse, os, pickle, sys, time
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn import model_selection, preprocessing
from scipy.stats import pearsonr, spearmanr
import numpy as np
import ml_rf__utils_updated_for_predictions as uti

# === Defaults ===
CV_N_SPLITS = 100
CV_TRAIN_SIZE = 0.8
CV_TEST_SIZE = 0.2
N_ESTIMATORS = 1000
MAX_FEATURES = 'sqrt'
SEED = 2121
METADATA_SEP_CHOICES = [',', ';', '\t']
VERSION = "1.0"

def read_params():
    """Parse command-line arguments."""
    p = argparse.ArgumentParser(description="Random Forest Regression with optional external prediction mode",
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # Required input
    p.add_argument('-i', '--inp_metaphlan', type=str, help="Path to Metaphlan abundance table")
    p.add_argument('-m', '--inp_metadata', type=str, help="Path to metadata table")
    p.add_argument('-c', '--inp_metadata_col', type=str, default=None, help="Column in metadata to match with feature table")
    p.add_argument('--inp_metadata_sep', default='\t', choices=METADATA_SEP_CHOICES, help="Metadata file separator")
    p.add_argument('-l', '--label', type=str, required=True, help="Target variable to predict")
    p.add_argument('-o', '--output', type=str, required=True, help="Output folder or prediction file")

    # External prediction mode
    p.add_argument('--model', type=str, default=None, help="Use existing model+normalizer to predict new data")

    # Taxa selection
    taxa_group = p.add_mutually_exclusive_group()
    taxa_group.add_argument('--taxa_all_taxa_levels', action='store_true', help="Use all taxa levels")
    taxa_group.add_argument('--taxa_only_SGBs', action='store_true', help="Use only SGBs")

    # Transformations
    tf_group = p.add_mutually_exclusive_group()
    tf_group.add_argument('--species_transabu_log', action='store_true', help="Log transform abundances")
    tf_group.add_argument('--species_transabu_arcsinsqrt', action='store_true', help="Arcsin-sqrt transform")
    tf_group.add_argument('--species_transabu_logit', action='store_true', help="Logit transform")

    # Cleaning options
    p.add_argument('--clean_bottom_percentile', type=int, default=None)
    p.add_argument('--clean_top_percentile', type=int, default=None)

    # Cross-validation and RF settings
    p.add_argument('--cv_n_splits', type=int, default=CV_N_SPLITS)
    p.add_argument('--cv_train_size', type=float, default=CV_TRAIN_SIZE)
    p.add_argument('--cv_test_size', type=float, default=CV_TEST_SIZE)
    p.add_argument('--rf_n_estimators', type=int, default=N_ESTIMATORS)
    p.add_argument('--rf_max_features', type=str, default=MAX_FEATURES)
    p.add_argument('--seed', type=int, default=SEED)

    # Runtime options
    p.add_argument('-t', '--threads', type=int, default=1)
    p.add_argument('--remove_from_samples', type=str, default=None)
    p.add_argument('--verbose', action='store_true')
    p.add_argument('-v', '--version', action='version', version=f'%(prog)s {VERSION}')

    return p.parse_args()

def main():
    args = read_params()

    if args.verbose:
        uti.info(f"\n▶️ Running Random Forest Regression (version {VERSION})\n")
        uti.info("Command: {}\n\n".format(' '.join(sys.argv)), init_new_line=True)

    # === External prediction mode ===
    if args.model:
        if args.verbose:
            uti.info(f"🔮 Loading model for external prediction: {args.model}")

        with open(args.model, 'rb') as f:
            full = pickle.load(f)
            model = full['model']
            norm = full['normalizer']
            features = [f[0] for f in full['feature_importance'][0]]

        metadata, _ = uti.read_metadata(args.inp_metadata, args.inp_metadata_sep, args.inp_metadata_col, args.label, verbose=args.verbose)

        data = uti.read_features(args.inp_metaphlan,
                                 taxa_all=args.taxa_all_taxa_levels,
                                 taxa_SGBs=args.taxa_only_SGBs,
                                 arcsinsqrt=args.species_transabu_arcsinsqrt,
                                 log=args.species_transabu_log,
                                 logit=args.species_transabu_logit,
                                 remove_from_samples=args.remove_from_samples,
                                 verbose=args.verbose)

        X, y, _, idx2smp, _ = uti.match(data, metadata, label=args.label, features_to_keep=features, verbose=args.verbose)
        preds = model.predict(X)
        preds_scaled = norm.predict(preprocessing.scale(preds).reshape(-1, 1)).T[0]

        with open(args.output, 'w') as f:
            f.write("sample_id\tobserved\tpredicted\n")
            for s, o, p in zip([idx2smp[i] for i in range(len(X))], y, preds_scaled):
                f.write(f"{s}\t{o}\t{p}\n")

        if args.verbose:
            uti.info(f"✅ Predictions written to {args.output}")
        return

    # === Training mode ===
    if args.verbose:
        uti.info("📚 Training model with cross-validation\n")

    metadata, _ = uti.read_metadata(args.inp_metadata, args.inp_metadata_sep, args.inp_metadata_col, args.label, verbose=args.verbose)

    data = uti.read_features(args.inp_metaphlan,
                             taxa_all=args.taxa_all_taxa_levels,
                             taxa_SGBs=args.taxa_only_SGBs,
                             arcsinsqrt=args.species_transabu_arcsinsqrt,
                             log=args.species_transabu_log,
                             logit=args.species_transabu_logit,
                             remove_from_samples=args.remove_from_samples,
                             verbose=args.verbose)

    X, y, feature_labels, index_2_sample, sample_2_index = uti.match(data, metadata, label=args.label,
                                                                     clean_bottom_percentile=args.clean_bottom_percentile,
                                                                     clean_top_percentile=args.clean_top_percentile,
                                                                     verbose=args.verbose)

    folds = model_selection.ShuffleSplit(n_splits=args.cv_n_splits,
                                         train_size=args.cv_train_size,
                                         test_size=args.cv_test_size,
                                         random_state=args.seed)

    results = {'predictions': [], 'pearson': [], 'spearman': [], 'feature_importance': []}
    output_base = os.path.join(args.output, args.label.replace('/', '_').replace(' ', '_'))

    for i, (train_idx, test_idx) in enumerate(folds.split(X)):
        train_X, train_y, test_X, test_y = uti.get_train_test(X, y, train_idx, test_idx, label=args.label,
                                                              idx2smp=index_2_sample,
                                                              smp2idx=sample_2_index,
                                                              get_pairs=None,
                                                              verbose=args.verbose)

        if args.verbose:
            uti.info(f'Fold {i:3d} | Train: {len(train_idx)} | Test: {len(test_idx)}')

        regr = RandomForestRegressor(n_estimators=args.rf_n_estimators,
                                     n_jobs=args.threads,
                                     random_state=args.seed,
                                     max_features=args.rf_max_features)

        norm = LinearRegression(n_jobs=args.threads)
        regr.fit(train_X, train_y)
        norm.fit(preprocessing.scale(regr.predict(train_X)).reshape(-1, 1), train_y.reshape(-1, 1))

        pred_y = regr.predict(test_X)
        pred_y_norm = norm.predict(preprocessing.scale(pred_y).reshape(-1, 1)).T[0]

        results['predictions'].append(list(zip((index_2_sample[i] for i in test_idx), test_y, pred_y_norm)))
        results['pearson'].append(pearsonr(test_y, pred_y_norm))
        results['spearman'].append(spearmanr(test_y, pred_y_norm))
        results['feature_importance'].append(list(zip(feature_labels, regr.feature_importances_)))

    # Save everything
    results['model'] = regr
    results['normalizer'] = norm

    with open(f"{output_base}.pkl", 'wb') as f:
        pickle.dump({args.label: results}, f, pickle.HIGHEST_PROTOCOL)

    with open(f"{output_base}_model_and_norm.pkl", 'wb') as f:
        pickle.dump({
            'model': regr,
            'normalizer': norm,
            'feature_importance': results['feature_importance']
        }, f, pickle.HIGHEST_PROTOCOL)

    if args.verbose:
        uti.info(f"✅ Full results saved to: {output_base}.pkl")
        uti.info(f"🧠 Model for external prediction saved to: {output_base}_model_and_norm.pkl")

if __name__ == '__main__':
    t0 = time.time()
    main()
    t1 = time.time()
    uti.info(f'⏱️ Total runtime: {int(t1 - t0)} seconds\n', init_new_line=True)
    sys.exit(0)
