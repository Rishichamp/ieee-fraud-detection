import shap
import matplotlib.pyplot as plt
import joblib

def explain_xgboost(model, X_test, feature_names):
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)

    # Global summary plot
    shap.summary_plot(shap_values, X_test,
                      feature_names=feature_names,
                      plot_type='bar', show=False)
    plt.savefig('shap_summary.png', bbox_inches='tight')
    plt.close()

    # Single transaction explanation
    def explain_single(idx):
        shap.waterfall_plot(
            shap.Explanation(
                values=shap_values[idx],
                base_values=explainer.expected_value,
                data=X_test[idx],
                feature_names=feature_names
            )
        )

    return shap_values, explain_single