#  Medical Insurance Cost Prediction

![Python](https://img.shields.io/badge/Python-3.9-blueviolet)
![Scikit-Learn](https://img.shields.io/badge/Library-Scikit_Learn-orange)
![XGBoost](https://img.shields.io/badge/Library-XGBoost-red)
![Status](https://img.shields.io/badge/Status-Completed-brightgreen)

##  Executive Summary
In the insurance industry, accurate pricing is critical for profitability and risk management. This project builds a machine learning pipeline to predict individual medical costs billed by health insurance.

**The Goal:** Predict insurance charges (`charges`) based on demographic and health factors (age, BMI, smoking status, etc.).

**The Result:** Developed an **XGBoost Regressor** model that explains **85%** of the variance in medical costs ($R^2 = 0.85$), significantly outperforming a baseline Linear Regression model ($R^2 = 0.40$).

##  Key Business Insights
Through rigorous Exploratory Data Analysis (EDA), we uncovered the hidden drivers of high costs:

1.  **The "Smoker" Multiplier:** Smoking is the single strongest predictor of high charges.
2.  **The Obesity Interaction:** High BMI (*Obesity*) alone leads to a moderate cost increase. However, the combination of **Obesity + Smoking** creates a massive spike in charges (non-linear interaction), which linear models fail to capture.
3.  **Age Progression:** Medical costs show a strong, linear increase with age, escalating rapidly after 40.
4.  **Cost Skewness:** The majority of customers incur low costs (<$10k), while a small "high-risk" tail generates charges exceeding $50k.

##  The Approach (Data Science Cycle)

### 1. Data Cleaning & Integrity
* **Anomaly Detection:** Identified and corrected erroneous negative values in `age` and `children` columns using absolute value transformations.
* **Type Conversion:** Parsed the `charges` column from string format (containing `$`) to float for analysis.
* **Standardization:** Cleaned inconsistent categorical labels in `region` and `sex`.

### 2. Exploratory Data Analysis (EDA)
* Discovered a heavy **Right-Skew** in the target variable.
* Visualized the "Smoker vs. Non-Smoker" cost gap using scatter plots and box plots.
* **Action:** Applied **Log-Transformation (`np.log1p`)** to the target variable to normalize the distribution for modeling.

### 3. Feature Engineering
* **Interaction Discovery:** Identified that `BMI` requires context (Smoker status) to be predictive.
* **Preprocessing:** Applied One-Hot Encoding for categorical data and Standard Scaling for numerical features to prevent bias.

### 4. Model Selection & Evaluation
We benchmarked three algorithms to find the best fit for this non-linear problem:

| Model | $R^2$ Score | RMSE (Error) | Verdict |
| :--- | :---: | :---: | :--- |
| **Linear Regression** | 0.40 | ~$9,033 | **Failed:** Could not capture the Smoker/BMI interaction. |
| **Random Forest** | 0.84 | ~$4,617 | **Excellent:** Captured non-linear logic well. |
| **XGBoost** | **0.85** | **~$4,571** | **🏆 Champion:** Optimized residuals for lowest error. |

*Note: The massive jump from 0.40 to 0.85 confirmed that insurance pricing is a non-linear problem.*

##  Business Recommendations
Based on the model's feature importance analysis:
1.  **Dynamic Risk Pricing:** Implement a "Risk Multiplier" specifically for the **Obese Smoker** segment, rather than treating BMI and Smoking as separate additive risks.
2.  **Smoking Cessation Program:** Invest in programs to convert smokers to non-smokers; this is the single most effective lever for reducing claims liability.
3.  **Preventative Health Credits:** Target younger customers (20s-30s) with high BMI for preventative interventions before they cross the "40s Threshold" where costs historically escalate.

##  Project Structure
* `data/`: Contains the raw insurance dataset.
* `notebooks/`: The Jupyter Notebook with full cleaning, EDA, and modeling code.
* `images/`: Saved visualizations (Feature Importance, Residual Plots).
* `models/`: Saved `xgboost_model.pkl` for deployment.

##  How to Run
```bash
# Clone the repository
git clone  https://github.com/OmarKhallouki/-Insurance-charge-prediction.git

# Install dependencies
pip install pandas numpy scikit-learn seaborn matplotlib xgboost

# Run the notebook
jupyter notebook
