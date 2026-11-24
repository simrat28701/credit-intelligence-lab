# Learning guide

## 1. Start with the decision

Imagine a team that can review only some existing credit-card accounts. It wants to catch high-risk accounts without overwhelming reviewers or inconveniencing customers who would repay. The labels tell us who defaulted next month. The models assign a probability of default (PD). The threshold translates the probability into a review flag, and the loss scenario adds the size of the financial exposure.

The unit of analysis is one cardholder. The six months of inputs precede the outcome month. The official workbook has an extra row of short column aliases, so pandas reads the second row as the header. The loader checks the 30,000 × 25 shape, column names, missing values, unique IDs and both binary label values. No customers are dropped. Credit bands reuse the notebook's lower-inclusive boundaries. For example, NT$50,000 belongs in the 50–<100k band. For descriptive repayment groups, positive codes indicate months late. The chart combines zero and negative codes as “no positive delay code” because those codes should not all be given the same precise account interpretation. The logistic model retains each distinct code as its own category.

## 2. Describe the portfolio before modeling

For any cohort, observed default rate = number of defaults ÷ number of accounts. Because the target is 0 or 1, its arithmetic mean is the same rate. The histogram counts accounts by limit interval. The median limit is the middle of the sorted balances, with the two middle values averaged for an even sample. It is less sensitive to very large limits than the mean.

The pandas charts group by one variable at a time. In DuckDB, `GROUP BY` buckets rows by credit band and September status together. `COUNT(*)` gives the denominator, `SUM(target)` counts defaults, `AVG(target)` gives the rate and `MEDIAN(LIMIT_BAL)` describes scale. Cross-segments can reveal patterns hidden in one-variable averages. Check the counts alongside the rates, especially after filtering. To rebuild an overall rate from subgroup rates, weight by each subgroup's customer count.

In the full sample, 22.12% defaulted. The rate rose from 13.83% for no positive September delay code to 69.55% for two or more months late. All five credit-limit bands showed progressively lower observed rates as limits increased. Earlier underwriting and customer characteristics can drive that pattern. Increasing a person's limit is a different question that requires a causal design. Age bands differ in customer mix and size. For example, the 60+ group's 28.32% is based on 339 people. These charts describe the sample and can motivate questions about allocation and monitoring.

## 3. Design the evaluation

A fixed 80/20 random split gives 24,000 training rows and 6,000 test rows. Stratification roughly preserves the default proportion in both sets. A fixed seed makes the assignment reproducible. Both models get the same rows and feature set. The 19 inputs comprise the credit limit, six repayment status fields, six bills and six payments. The target, account ID and demographics stay outside the model. Age remains available for descriptive charts.

Leakage occurs when training uses information unavailable at decision time. Feature timing matters even if a field predicts well. Here the variables reflect the six preceding months. Scalers and category encoders learn from training rows inside the logistic pipeline. The test set supplies only predictions and evaluation. This random split evaluates generalization within the historical cohort. A later-period holdout would be needed to evaluate drift. Looking repeatedly at the holdout to pick a model or cutoff turns it into a development set.

## 4. Logistic regression

Logistic regression estimates $p = 1/(1+e^{-z})$, where $z = \beta_0 + \sum_j \beta_j x_j$. The log of the odds, $\log[p/(1-p)]$, equals $z$. Training minimizes the negative binary log-likelihood, plus the default L2 penalty, which shrinks coefficients and helps stabilize estimates when inputs overlap. A wrong, highly confident prediction incurs a larger log loss than a cautious prediction.

The scaler uses $(x-\mu_{train})/s_{train}$ with the training mean and population standard deviation. One-hot encoding creates a 0/1 indicator for each observed repayment code except −1, the reference. For an amount, $e^{\beta_j}$ multiplies modeled odds for a one-training-standard-deviation increase with the other terms fixed. For a code, it compares odds with the −1 code for that month. For example, a hypothetical coefficient of 0.69 roughly doubles odds. A starting probability of 20% gives odds of 0.25. Doubling those odds gives 0.50, or about 33% probability. Coefficients describe conditional model relationships. Overlapping monthly signals and regularization make causal or precise economic interpretations inappropriate.

This is the main explanatory benchmark because it yields probabilities with a compact, signed explanation. The driver chart selects the 12 largest absolute coefficients. It uses the sign to show direction.

## 5. Random forest

Each of 160 decision trees trains on a bootstrap sample drawn with replacement from the training set. At each node it considers a random subset of features and chooses a split that improves class purity. With default Gini impurity, a node with class shares $p_k$ has impurity $1-\sum_k p_k^2$. The chosen split reduces the weighted impurity of the children. A tree estimates PD from the observed class share in the leaf, and the forest averages the trees. Depth 10 and a minimum of 20 observations per leaf limit highly specific splits. The forest uses raw numeric amounts and repayment codes, so splits can group adjacent codes. Logistic regression treats those codes as separate categories. The two models therefore share raw inputs while transforming them differently.

A forest can capture non-linear effects and interactions. For example, the significance of a bill amount can depend on recent delays. The impurity importance chart sums weighted decreases attributable to splits and normalizes across features. Importance has no direction and may favor variables with many possible splits or distribute credit across correlated fields. The forest's holdout AUC of 0.774 exceeded logistic regression's 0.755 on this split. This comparison has no estimate of sampling uncertainty.

## 6. Read classification and probability metrics

At a threshold, flag if PD ≥ threshold. TP means a default was flagged, FP means a non-default was flagged, FN means a default was missed and TN means a non-default was left unflagged.

| Measure | Calculation and question |
|---|---|
| Accuracy | (TP + TN) ÷ N. How often was the classification correct? |
| Precision | TP ÷ (TP + FP). Of flagged accounts, how many defaulted? The app uses zero if there are no flags. |
| Recall | TP ÷ (TP + FN). Of actual defaults, how many were flagged? |
| ROC-AUC | Probability that a random default ranks above a random non-default, with ties counted as half. How well are accounts ordered? |
| Brier score | Mean of (predicted PD − binary outcome)². How close are probabilities to the outcomes in squared-error terms? |

Always classifying as non-default yields 77.88% test accuracy and 0% recall. Both trained models exceed 81% accuracy at 50%, yet their recall is around 35%. A team with a capacity constraint needs to know how many flags and missed defaults it will face directly.

Calibration asks whether predicted probabilities agree with observed frequencies. Order accounts by score and group into up to ten equal-frequency bins. Plot mean predicted PD against the fraction that defaulted. Above the diagonal, the model underpredicts in that bin. Below it, it overpredicts. Ranking could stay identical after a monotone score transformation while calibration changes. Brier includes contributions from probability calibration and resolution. Here it is 0.1392 for logistic regression and 0.1356 for random forest. The chart evaluates raw probabilities. If recalibration were needed, fit a calibrator on separate validation predictions or cross-validation folds within training, then assess it on untouched test data.

## 7. Connect the cutoff to expected credit loss

At a 30% cutoff the logistic model flags 1,129 accounts, including 628 actual defaulters and 501 false positives. It misses 699 defaults. At 20% it flags 1,672 accounts, catches 776 defaults and produces 896 false positives. The additional 543 reviews find 148 additional defaults. If review is time-consuming or disruptive, quantify those costs alongside the benefit of reaching a risky account.

Per account, $ECL_i = PD_i \times LGD \times EAD_i$. PD is the model's next-month probability. LGD is the assumed share of exposure lost if default occurs. EAD is the positive September statement balance multiplied by the exposure percentage. A negative statement balance is floored at zero. For PD 30%, LGD 45% and EAD NT$10,000, illustrative ECL is NT$1,350. Portfolio ECL adds all account estimates. An EAD share of 100% uses the full positive statement balance. A share above 100% is a sensitivity assumption about more exposure. It uses no observed future drawdown.

Changing the cutoff moves ECL between flagged and unflagged groups. Changing LGD or the EAD share scales the loss estimates and leaves flags unchanged. Flagged ECL quantifies risk represented in the review queue. Estimating avoidable loss also requires the effect of a review and may require customer-level variation in recovery rates and exposure. The public data contains none of those outcomes. New-loan approval additionally needs data available at application time and evidence about the selection created by prior approvals.

## 8. Follow the code

| File | What to trace |
|---|---|
| `notebooks/01_exploration.ipynb` | Original source inspection, baseline default rate and credit-limit band design. |
| `src/data.py` | UCI archive → validated pandas frame → descriptive bands. Also registers the filtered frame with DuckDB and calls the SQL file. |
| `sql/portfolio_segments.sql` | Counts, rates and median limits by two business dimensions. |
| `src/modeling.py` | Split → fit → aligned test probabilities → confusion counts, metrics, calibration points, driver values and account losses. |
| `src/pipeline.py` | Reproduces headline numbers, verifies SQL totals and writes the JSON results. |
| `app.py` | Streamlit navigation, visual layout and caching. The first overview requires only data. Opening a model page triggers fitting. |
| `src/dashboard.py` | Links widgets to filters and scores and wraps them in Plotly charts and KPI cards. |
| `.streamlit/config.toml` | Controls the Cyber Lavender theme and local telemetry preference. |

Try explaining the dashboard from a printed row. If the account's PD is 25%, it is unflagged at 30% and flagged at 20%. Its account-level ECL is identical in both cases. Explain why it would still take information about outcomes after intervention to claim the lower threshold reduces realized losses.

