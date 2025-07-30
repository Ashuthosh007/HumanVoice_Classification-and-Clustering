import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st


def label_distribution(df):
    sns.countplot(x="label", data=df)
    plt.title("Label Distribution (0 = Female, 1 = Male)")
    plt.xlabel("Gender")
    plt.ylabel("Count")
    st.pyplot(plt.gcf())
    plt.clf()


def feature_correlation(df):
    numeric_df = df.select_dtypes(include=['float64', 'int64']) 
    corr = numeric_df.corr()
    plt.figure(figsize=(12, 10))
    sns.heatmap(corr, cmap="coolwarm", annot=False)
    plt.title("Feature Correlation Heatmap")
    st.pyplot(plt.gcf())
    plt.clf()

def show_feature_importance(model, feature_names, top_n=20):
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
        indices = importances.argsort()[::-1][:top_n]
        
        plt.figure(figsize=(10, 6))
        sns.barplot(
            x=importances[indices],
            y=[feature_names[i] for i in indices],
            palette="viridis"
        )
        plt.title("Top Feature Importances")
        plt.xlabel("Importance Score")
        plt.ylabel("Features")
        st.pyplot(plt.gcf())
        plt.clf()
    else:
        st.warning("Feature importances not available for this model.")