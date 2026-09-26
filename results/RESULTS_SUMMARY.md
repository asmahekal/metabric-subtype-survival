# Results summary (auto-generated)

Run: 2026-09-24 23:00 | runtime 163.5 min | 3x5-fold repeated stratified CV | seed 42 | quick=False

Cohort: n = 1902 | OS events = 1102 | breast-cancer deaths = 622 | other-cause deaths = 480 | median follow-up = 115.6 months

## Subgroups

| subgroup                   |    n |   os_events |   dss_events |   other_cause_deaths |   median_follow_up_months |   os_5y_KM |   dss_5y_KM |   os_10y_KM |   dss_10y_KM |
|:---------------------------|-----:|------------:|-------------:|---------------------:|--------------------------:|-----------:|------------:|------------:|-------------:|
| All                        | 1902 |        1102 |          622 |                  480 |                   115.617 |      0.779 |       0.822 |       0.588 |        0.709 |
| TNBC                       |  298 |         161 |          113 |                   48 |                    91.767 |      0.659 |       0.687 |       0.535 |        0.636 |
| Basal                      |  199 |         111 |           81 |                   30 |                    89.033 |      0.612 |       0.638 |       0.503 |        0.6   |
| HER2-enriched              |  220 |         155 |          107 |                   48 |                    95.783 |      0.644 |       0.673 |       0.467 |        0.534 |
| Claudin-low                |  198 |          89 |           56 |                   33 |                   116.183 |      0.807 |       0.822 |       0.664 |        0.745 |
| Aggressive (Basal/HER2/CL) |  617 |         355 |          244 |                  111 |                   102.033 |      0.685 |       0.709 |       0.54  |        0.622 |
| LumB                       |  461 |         303 |          181 |                  122 |                   104.467 |      0.76  |       0.806 |       0.509 |        0.636 |
| LumA                       |  678 |         363 |          146 |                  217 |                   128.95  |      0.875 |       0.929 |       0.687 |        0.834 |

## OS

* Best global configuration (whole cohort): **RSF [Clin+Genes+Mut]**
  * Uno C = 0.701 (95% CI 0.687–0.715)
  * Harrell C = 0.692 (95% CI 0.680–0.704)
  * AUC(5y) = 0.757 (95% CI 0.736–0.778)
  * IBS = 0.172 (95% CI 0.167–0.176)
  * 5-y binary: balanced accuracy = 0.680 (95% CI 0.657–0.702), MCC = 0.315 (95% CI 0.271–0.358)
* NPI alone: Uno C = 0.643 (95% CI 0.630–0.657)
* Clinical CoxPH: Uno C = 0.680 (95% CI 0.665–0.695)

### Subtype-specific vs global (best model per subgroup & feature set, ΔUno C)

* Aggressive (Basal/HER2/CL) | Clin+Genes | RSF: global 0.675 vs specific 0.667; Δ = -0.008 (95% CI -0.025 to +0.008), p = 0.303, specific wins 33% of folds
* Aggressive (Basal/HER2/CL) | Clinical | RSF: global 0.675 vs specific 0.677; Δ = +0.003 (95% CI -0.006 to +0.012), p = 0.534, specific wins 67% of folds
* Basal | Clin+Genes | RSF: global 0.595 vs specific 0.455; Δ = -0.140 (95% CI -0.222 to -0.058), p = 0.00251, specific wins 0% of folds
* Basal | Clinical | XGB-Cox: global 0.603 vs specific 0.598; Δ = -0.005 (95% CI -0.076 to +0.066), p = 0.873, specific wins 53% of folds
* Claudin-low | Clin+Genes | GBSA: global 0.738 vs specific 0.598; Δ = -0.140 (95% CI -0.206 to -0.074), p = 0.000433, specific wins 0% of folds
* Claudin-low | Clinical | RSF: global 0.725 vs specific 0.690; Δ = -0.034 (95% CI -0.093 to +0.024), p = 0.229, specific wins 13% of folds
* HER2-enriched | Clin+Genes | RSF: global 0.665 vs specific 0.611; Δ = -0.055 (95% CI -0.105 to -0.004), p = 0.0363, specific wins 7% of folds
* HER2-enriched | Clinical | RSF: global 0.676 vs specific 0.665; Δ = -0.011 (95% CI -0.056 to +0.034), p = 0.61, specific wins 40% of folds
* LumA | Clin+Genes | XGB-Cox: global 0.730 vs specific 0.716; Δ = -0.015 (95% CI -0.040 to +0.011), p = 0.235, specific wins 33% of folds
* LumA | Clinical | GBSA: global 0.730 vs specific 0.738; Δ = +0.009 (95% CI -0.008 to +0.026), p = 0.292, specific wins 60% of folds
* LumB | Clin+Genes | RSF: global 0.655 vs specific 0.658; Δ = +0.002 (95% CI -0.027 to +0.031), p = 0.863, specific wins 47% of folds
* LumB | Clinical | XGB-Cox: global 0.654 vs specific 0.635; Δ = -0.019 (95% CI -0.042 to +0.005), p = 0.11, specific wins 13% of folds
* TNBC | Clin+Genes | RSF: global 0.671 vs specific 0.635; Δ = -0.036 (95% CI -0.094 to +0.021), p = 0.195, specific wins 13% of folds
* TNBC | Clinical | RSF: global 0.665 vs specific 0.666; Δ = +0.001 (95% CI -0.026 to +0.028), p = 0.937, specific wins 53% of folds

* Top SHAP features – Global model | All: age_at_diagnosis, nottingham_prognostic_index, lymph_nodes_examined_positive, tumor_size, stat5a, map2k4, vegfa, type_of_breast_surgery_BREAST CONSERVING, flt3, gata3_mut
* Top SHAP features – Specific | TNBC: age_at_diagnosis, e2f6, lymph_nodes_examined_positive, dlec1, rbpjl, nottingham_prognostic_index, kdm3a, kdr, nr3c1, cyp17a1
* Top SHAP features – Specific | Basal: ugt2b17, map3k4, smad7, flt3, acvr1b, cul1, shbg, hsd3b2, prkcz, nottingham_prognostic_index
* Top SHAP features – Specific | HER2-enriched: lymph_nodes_examined_positive, nottingham_prognostic_index, acvr2b, dll4, rbpj, itch, map2, eif5a2, prkacg, cdkn1a
* Top SHAP features – Specific | Claudin-low: age_at_diagnosis, rps6, ttyh1, nottingham_prognostic_index, lymph_nodes_examined_positive, akap9, mmp13, stat5b, shbg, kit
* Top SHAP features – Specific | Aggressive (Basal/HER2/CL): lymph_nodes_examined_positive, age_at_diagnosis, tumor_size, shbg, nottingham_prognostic_index, rbpj, ptpn22, arrdc1, csf1r, rab25
* Top SHAP features – Specific | LumB: age_at_diagnosis, lymph_nodes_examined_positive, tumor_size, nottingham_prognostic_index, bmf, flt3, dll3, acvr1b, nr2f1, peg3
* Top SHAP features – Specific | LumA: age_at_diagnosis, acvr2a, type_of_breast_surgery_BREAST CONSERVING, nottingham_prognostic_index, ctcf, stat5a, tumor_size, her2_status_measured_by_snp6_GAIN, gsk3b, rps6ka2

## DSS

* Best global configuration (whole cohort): **RSF [Clin+Genes]**
  * Uno C = 0.734 (95% CI 0.720–0.749)
  * Harrell C = 0.732 (95% CI 0.720–0.745)
  * AUC(5y) = 0.804 (95% CI 0.783–0.825)
  * IBS = 0.122 (95% CI 0.115–0.129)
  * 5-y binary: balanced accuracy = 0.721 (95% CI 0.700–0.743), MCC = 0.379 (95% CI 0.348–0.410)
* NPI alone: Uno C = 0.684 (95% CI 0.666–0.701)
* Clinical CoxPH: Uno C = 0.707 (95% CI 0.691–0.722)

### Subtype-specific vs global (best model per subgroup & feature set, ΔUno C)

* Aggressive (Basal/HER2/CL) | Clin+Genes | RSF: global 0.688 vs specific 0.673; Δ = -0.015 (95% CI -0.033 to +0.003), p = 0.0951, specific wins 13% of folds
* Aggressive (Basal/HER2/CL) | Clinical | RSF: global 0.680 vs specific 0.682; Δ = +0.002 (95% CI -0.011 to +0.015), p = 0.724, specific wins 47% of folds
* Basal | Clin+Genes | RSF: global 0.621 vs specific 0.464; Δ = -0.157 (95% CI -0.240 to -0.074), p = 0.00118, specific wins 0% of folds
* Basal | Clinical | RSF: global 0.626 vs specific 0.611; Δ = -0.015 (95% CI -0.052 to +0.021), p = 0.382, specific wins 20% of folds
* Claudin-low | Clin+Genes | GBSA: global 0.740 vs specific 0.604; Δ = -0.136 (95% CI -0.220 to -0.052), p = 0.00368, specific wins 0% of folds
* Claudin-low | Clinical | RSF: global 0.707 vs specific 0.662; Δ = -0.045 (95% CI -0.120 to +0.030), p = 0.223, specific wins 27% of folds
* HER2-enriched | Clin+Genes | RSF: global 0.678 vs specific 0.616; Δ = -0.062 (95% CI -0.105 to -0.018), p = 0.00879, specific wins 7% of folds
* HER2-enriched | Clinical | XGB-Cox: global 0.686 vs specific 0.641; Δ = -0.045 (95% CI -0.088 to -0.001), p = 0.0459, specific wins 0% of folds
* LumA | Clin+Genes | GBSA: global 0.748 vs specific 0.708; Δ = -0.040 (95% CI -0.097 to +0.017), p = 0.153, specific wins 13% of folds
* LumA | Clinical | RSF: global 0.709 vs specific 0.697; Δ = -0.012 (95% CI -0.041 to +0.016), p = 0.375, specific wins 27% of folds
* LumB | Clin+Genes | CW-GBSA: global 0.695 vs specific 0.652; Δ = -0.043 (95% CI -0.086 to -0.000), p = 0.0494, specific wins 13% of folds
* LumB | Clinical | GBSA: global 0.683 vs specific 0.662; Δ = -0.021 (95% CI -0.058 to +0.016), p = 0.243, specific wins 33% of folds
* TNBC | Clin+Genes | RSF: global 0.669 vs specific 0.646; Δ = -0.023 (95% CI -0.066 to +0.020), p = 0.268, specific wins 27% of folds
* TNBC | Clinical | RSF: global 0.666 vs specific 0.655; Δ = -0.011 (95% CI -0.028 to +0.006), p = 0.175, specific wins 20% of folds

* Top SHAP features – Global model | All: lymph_nodes_examined_positive, nottingham_prognostic_index, stat5a, tumor_size, aurka, mlh1, age_at_diagnosis, pik3ca_mut, diras3, hes6
* Top SHAP features – Specific | TNBC: lymph_nodes_examined_positive, dtx3, nottingham_prognostic_index, e2f6, cyp21a2, kdr, hes1, stat5b, kdm3a, mmp1
* Top SHAP features – Specific | Basal: map3k4, ugt2b17, nottingham_prognostic_index, cul1, acvr2b, hes1, bmp6, flt3, hsd3b2, type_of_breast_surgery_BREAST CONSERVING
* Top SHAP features – Specific | HER2-enriched: lymph_nodes_examined_positive, smad1, abcc1, rbpj, nottingham_prognostic_index, map2, smad6, dll4, cdkn1a, mmp16
* Top SHAP features – Specific | Claudin-low: frmd3, shbg, heyl, lymph_nodes_examined_positive, shank2, akap9, stat5b, ppp2r2a, tubb4a, tgfb2
* Top SHAP features – Specific | Aggressive (Basal/HER2/CL): lymph_nodes_examined_positive, nottingham_prognostic_index, rbpj, shbg, men1, stat5b, cxcl8, mapt, rab25, jak2
* Top SHAP features – Specific | LumB: nottingham_prognostic_index, lymph_nodes_examined_positive, acvr1b, nr2f1, dph1, stat2, akap9, tumor_size, foxo3, pik3ca_mut
* Top SHAP features – Specific | LumA: pms2, stat5a, apaf1, stat5b, age_at_diagnosis, diras3, tumor_size, nottingham_prognostic_index, cxcr1, lymph_nodes_examined_positive

## Notes for the manuscript

* All preprocessing (imputation, scaling, one-hot encoding, mutation filtering, univariate gene selection) and hyper-parameter tuning (CoxNet inner CV) were fitted inside training folds only.
* Global and subtype-specific models were evaluated on the identical held-out patients of each fold; paired differences were tested with the Nadeau–Bengio corrected resampled t-test and Wilcoxon signed-rank test, with Benjamini–Hochberg FDR (q_bh_uno_c).
* 5-year binary metrics exclude patients censored before 60 months; thresholds were chosen by Youden's J on training folds.
* Pooled out-of-fold figures (KM tertiles, calibration, DCA, ROC) use the configuration with the best CV Uno C – report fold-level CV metrics (T06/T07) as the primary, unbiased estimates.
* SHAP models were fitted on the full (sub)cohort for explanation only; they are not used for performance estimates.