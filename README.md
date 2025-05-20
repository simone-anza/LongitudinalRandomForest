# LongitudinalRandomForest
LongitudinalRandomForest: A novel framework for analyzing compositional data in longitudinal studies using Random Forest models. Maintains sample independence through intelligent randomization, ensuring one sample per subject per iteration while utilizing all available data across runs. Ideal for compositional data analysis challenges.

##LongitudinalRandomForest
A robust framework for analyzing longitudinal compositional data using randomized sampling and Random Forest models.

###Overview
LongitudinalRandomForest is a specialized tool designed to analyze longitudinal compositional data (such as microbiome data) while addressing the challenges of repeated measures and subject independence. The framework uses a novel randomization approach to ensure that each analysis iteration uses exactly one sample per subject, maximizing data independence while utilizing all available samples over multiple iterations.
Key Features

Subject-Independent Sampling: Ensures each analysis run contains only one sample per subject
Complete Sample Coverage: All samples appear at least once across iterations
Compositional Data Analysis: Specifically designed for microbiome and other compositional data types
Random Forest Implementation: Leverages both classification and regression capabilities
Cross-Validation: Implements robust 10×10-fold cross-validation

###Methodology
Sample Randomization Process
For longitudinal datasets with multiple samples per subject (e.g., 5 samples per subject):

The orchestrator creates N datasets (where N ≥ number of samples per subject)
Each dataset contains exactly one sample per subject
Different samples are selected for each subject across repetitions
This ensures no repeated samples per subject within a dataset
All samples appear at least once across all datasets

##Analysis Workflow
The framework categorizes data based on a variable of interest (e.g., maternal cytokines):

Category 0: Middle values (25-75% of distribution)
Category 1: Extreme values (bottom 25% and top 75% of distribution)

For each category:

Use compositional data (e.g., microbial abundances) to predict an outcome (e.g., age)
Apply the randomization approach to ensure one sample per subject
Perform multiple repetitions of 10-fold cross-validation
Compare predictions between categories to assess impact of the variable of interest

##Use Case
The primary use case is investigating how a variable of interest (such as maternal cytokines during pregnancy) might influence another outcome (such as biological aging) as measured through compositional data (like gut microbiome profiles).
Requirements

Python 3.7+
scikit-learn
numpy
pandas
bash (for orchestration)

#Installation
bashgit clone https://github.com/yourusername/LongitudinalRandomForest.git
cd LongitudinalRandomForest
pip install -r requirements.txt
Usage
Basic Usage
bash./run_analysis.sh --input_data your_data.csv --subject_col subject_id --timepoint_col visit --outcome age --category_var maternal_cytokines
Configuration
Create a config file to customize analysis parameters:
json{
  "n_iterations": 10,
  "n_folds": 10,
  "random_state": 42,
  "min_samples_leaf": 5,
  "n_estimators": 500
}
Then run with:
bash./run_analysis.sh --config my_config.json --input_data your_data.csv
Output
The framework produces:

Performance metrics for each category
Comparison statistics between categories
Visualizations of prediction performance
Feature importance rankings

Citation
If you use this framework in your research, please cite:
Simone Anzà (2025). LongitudinalRandomForest: A Framework for Analyzing Longitudinal Compositional Data Using Randomized Sampling and Random Forest Models

#License
This project is licensed under the MIT License - see the LICENSE file for details.
Contributing
Contributions are welcome! Please feel free to submit a Pull Request.
