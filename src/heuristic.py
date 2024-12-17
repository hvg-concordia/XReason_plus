from scipy import stats

def calculate_kendall_tau(tool1_selection, tool2_ranks,total_features):
    # Assign equal ranks to features selected by Tool 1
    tool1_ranks = []
    for i in range(total_features):
        if i in tool1_selection:
            tool1_ranks.append(1)  # Assign rank 1 to all selected features
        else:
            tool1_ranks.append(len(tool1_selection) + 1)  # Assign a lower rank to non-selected features

    # Calculate Kendall's tau
    tau, p_value = stats.kendalltau(tool1_ranks, tool2_ranks)
    return tau,p_value


def calculate_spearman_correlation(tool1_selection, tool2_ranks,total_features):
    # Assign equal ranks to features selected by Tool 1
    tool1_ranks = []
    for i in range(total_features):
        if i in tool1_selection:
            tool1_ranks.append(1)  # Assign rank 1 to all selected features
        else:
            tool1_ranks.append(len(tool1_selection) + 1)  # Assign a lower rank to non-selected features

    # Calculate Spearman's rank correlation
    correlation, p_value = stats.spearmanr(tool1_ranks, tool2_ranks)
    return correlation,p_value

def shap_to_rankings(shap_values_sample):
    # Get absolute SHAP values
    abs_shap = np.abs(shap_values_sample)

    # Create a DataFrame with feature names and their absolute SHAP values
    feature_importance = pd.DataFrame({
        'feature': X_test.columns,
        'importance': abs_shap
    })

    # Sort by importance (higher absolute SHAP value = more important)
    feature_importance = feature_importance.sort_values('importance', ascending=False)

    # Return the list of features in order of importance
    return list(X_test.columns.get_indexer(feature_importance['feature'].tolist()))

