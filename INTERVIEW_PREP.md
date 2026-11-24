# Interview preparation

## 30-second pitch

Credit Intelligence Lab shows how a card issuer could turn default probabilities into an account-review policy. I analyzed 30,000 historical Taiwanese cardholders, used SQL to segment risk, and compared logistic regression with a random forest on a common holdout. Then I built a Streamlit simulator that shows how a lower threshold catches more defaults, adds reviews and changes the amount of modeled loss in the review queue. The project separates what the data measures from the exposure and recovery assumptions that drive the loss scenario.

## Three-minute demonstration

| Time | What to show and say |
|---|---|
| 0:00–0:40 | Open **Portfolio Overview**. Point out 30,000 customers and 22.12% default. Compare the September repayment groups and use one credit-limit filter. Show the SQL table and identify the denominator for each rate. Clear the filter. |
| 0:40–1:25 | Open **Default Prediction**. Cite AUC of 0.755 and 0.774. Compare the 77.88% majority accuracy with zero recall. Show the confusion matrices and explain why 50% is an example cutoff. Point to a calibration bin, then switch between the signed logistic coefficients and forest importance. |
| 1:25–2:30 | Open **Credit Risk Simulator**. Select logistic regression, 30% cutoff, 45% LGD and 100% EAD share. Lower the cutoff to 20%. Explain that 543 more reviews catch 148 additional defaults and add 395 false positives. Show how flagged ECL changes while the total stays constant. Adjust LGD and point out the impact on loss estimates. |
| 2:30–3:00 | Open **Findings and Limitations**. Explain the historical cohort, absence of rejected applicants and unobserved intervention effects. Close with the practical decision the simulator supports: how much review capacity to allocate at different default-risk cutoffs. |

## 15 recruiter questions and answers

1. **What business problem did you tackle?** I quantified the workload, missed defaults and modeled loss exposure associated with reviewing existing card accounts at different risk thresholds.
2. **Why choose this dataset?** UCI offers an attributed source with 30,000 customers, several months of account history and a clearly defined next-month outcome.
3. **What did you reuse?** I used the original workbook download and the five credit-limit bands from my exploratory notebook. The portfolio pipeline recalculates the counts and rates.
4. **What did SQL add?** DuckDB calculates counts, defaults, rates and the median credit limit across joint segments inside the filtered cohort.
5. **What was your clearest descriptive finding?** The observed rate was 69.55% with a September delay of at least two months and 13.83% with no positive delay code.
6. **Why logistic regression?** It gives a probability and signed coefficients. I can explain a change in odds conditional on the model's other inputs.
7. **Why random forest?** It captures non-linear interactions and gives me a practical benchmark against the simpler model.
8. **Which model performed better?** The forest had higher AUC and lower Brier loss on the shared holdout. Logistic regression is easier to explain. Model choice depends on the operating point and validation evidence.
9. **How did you handle class imbalance?** I preserved the observed prevalence with stratification and reported precision, recall, confusion counts and the majority baseline alongside accuracy.
10. **How did you avoid leakage?** I restricted inputs to earlier account history and fitted the logistic encoder and scaler on training accounts. IDs and outcome are excluded.
11. **Why does calibration matter?** Multiplying PD by an exposure assumes PD has a meaningful scale. A calibrated 20% bucket should default about 20% of the time in the evaluation population.
12. **What happens at a lower threshold?** A larger review queue catches more defaults and includes more customers who would repay. At 20%, the logistic model flags 1,672 versus 1,129 at 30%.
13. **What does ECL mean in the app?** It is a one-month scenario estimate using model PD, assumed severity and statement balance as an exposure proxy. It quantifies risk represented in the reviewed and unreviewed accounts.
14. **How did you address fairness?** I omitted demographic inputs from both models and discuss residual proxy effects. Representative subgroup calibration, error rates and support counts would be part of a fairness audit.
15. **Can a bank use this to approve a new loan?** The sample is historical and conditional on already holding a card. Applicant inputs and outcomes under different approval policies require separate validation.

## Five challenging technical questions

1. **Could a model have the same AUC but different ECL?** Yes. A strictly increasing transformation preserves ranking and therefore AUC, but it changes the probabilities multiplied by LGD and EAD. It can also change flags at a fixed numerical cutoff.
2. **How would you choose a cutoff in production?** Use validation data, observed review costs, an estimate of intervention effectiveness, and a capacity constraint. Compare feasible policies on an untouched later-period sample. PD-only ranking can be suboptimal if exposure or preventable loss varies across accounts.
3. **If an intervention always prevents a fixed share of ECL, how would you prioritize review?** Under that assumed effect, compare preventable ECL minus review and customer costs for each account. Capacity can be allocated by expected net benefit when reviews consume equal resources. The dataset does not estimate the assumed effect.
4. **Why could an important forest variable have a small logistic coefficient?** The forest can split on non-linearities and interactions. Correlated inputs redistribute model effects, and impurity importance measures split improvements while a signed logistic coefficient measures conditional change in log odds.
5. **What would you change with more data?** Use a temporal split across multiple cohorts, check calibration drift and subgroup error rates, and obtain actual balances at default, recoveries and intervention outcomes before estimating policy value.

