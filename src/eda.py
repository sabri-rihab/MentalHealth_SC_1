import pandas as pd
import matplotlib as plt
import seaborn as sns


def plot_class_distribution(df):
    plt.figure(figsize=(7, 4))
    sns.countplot(
        data=df,
        x='Mental_Health_Risk',
        order=df['Mental_Health_Risk'].value_counts().index,
        palette='Set2'
    )

    plt.title('Mental Health Risk - Class Distribution')
    plt.xlabel('Class')
    plt.ylabel('Count')

    plt.tight_layout()
    plt.show()

def compute_correlations(df):
    num_cols = df.select_dtypes(exclude=['object', 'string'])

    # df_encoded = pd.get_dummies(df[num_cols], drop_first=False)
    df_encoded = pd.get_dummies(df[num_cols + ['Mental_Health_Risk']], columns=['Mental_Health_Risk'], drop_first=False, dtype=int)
    corr = df_encoded.corr()
    print(corr)


def plot_correlation_heatmap():
def plot_categorical_vs_target():
def top_target_correlations():