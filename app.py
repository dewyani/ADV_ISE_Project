import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import openai
from sklearn.preprocessing import LabelEncoder

# Set up OpenAI API key
openai.api_key = 'key'

def generate_visualizations(df):
    """Automatically generate key visualizations based on the dataset."""
    st.header("Generated Visualizations")

    # Detect numerical and categorical columns
    num_cols = df.select_dtypes(include=['int64', 'float64']).columns
    cat_cols = df.select_dtypes(include=['object', 'category']).columns

    # Visualization 1: Correlation heatmap for numerical columns
    if len(num_cols) > 1:
        st.subheader("Correlation Heatmap")
        plt.figure(figsize=(10, 6))
        sns.heatmap(df[num_cols].corr(), annot=True, cmap="coolwarm", fmt=".2f")
        st.pyplot(plt)

    # Visualization 2: Countplot for categorical columns
    for col in cat_cols:
        st.subheader(f"Countplot for {col}")
        plt.figure(figsize=(8, 4))
        sns.countplot(data=df, x=col)
        plt.xticks(rotation=45)
        st.pyplot(plt)

    # Visualization 3: Boxplots for numerical vs categorical columns
    for num_col in num_cols:
        for cat_col in cat_cols:
            st.subheader(f"Boxplot: {num_col} vs {cat_col}")
            plt.figure(figsize=(8, 4))
            sns.boxplot(data=df, x=cat_col, y=num_col)
            plt.xticks(rotation=45)
            st.pyplot(plt)

    # Visualization 4: Histogram for numerical columns
    for col in num_cols:
        st.subheader(f"Histogram for {col}")
        plt.figure(figsize=(8, 4))
        sns.histplot(df[col], kde=True, bins=30)
        st.pyplot(plt)

def process_nl_query(query, df):
    """Process natural language query and generate appropriate visualization."""
    response = openai.ChatCompletion.create(
    model="gpt-3.5-turbo",
    messages=[
        {"role": "system", "content": "You are a data analysis assistant."},
        {"role": "user", "content": f"Dataset columns: {list(df.columns)}. Query: {query}"}
    ],
    max_tokens=150  # Reduce response length
)

# Extract the response text
    response_text = response['choices'][0]['message']['content']


    # Extract column names and visualization type from GPT response
    generated_text = response['choices'][0]['message']['content']
    st.write("**GPT Response:**", generated_text)

    # Example: Extract column and chart suggestions using basic parsing (customize as needed)
    if "bar chart" in generated_text.lower() and "Gender" in query:
        plt.figure(figsize=(8, 4))
        sns.countplot(data=df, x="Gender")
        st.pyplot(plt)
    elif "boxplot" in generated_text.lower() and "marks" in query.lower():
        plt.figure(figsize=(8, 4))
        sns.boxplot(data=df, x="Gender", y="Marks")
        st.pyplot(plt)
    else:
        st.write("Could not process query. Please try rephrasing.")

# Streamlit interface
st.title("Automated Data Visualization System")

uploaded_file = st.file_uploader("Upload your dataset (CSV)", type="csv")

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write("### Preview of the Dataset")
    st.write(df.head())

    # Generate automated visualizations
    generate_visualizations(df)

    # Allow user to enter natural language queries
    st.header("Natural Language Query")
    query = st.text_input("Ask something about the dataset:")

    if st.button("Generate Visualization"):
        process_nl_query(query, df)
