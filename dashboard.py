"""
KARMIX TECH INTERNSHIP
Project  : Customer Churn Prediction Dashboard
Step 4   : Streamlit Dashboard
"""

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


st.set_page_config(
    page_title="Customer Churn Dashboard",
    page_icon="📊",
    layout="wide"
)


@st.cache_data
def load_data():
    df = pd.read_csv(r"D:\Karmix_Tech\Customer_churn_prediction\Clean_datasets\telco_churn_cleaned.csv")
    return df

df = load_data()


st.sidebar.title("🔍 Filters")

contract_filter = st.sidebar.multiselect(
    "Contract Type",
    options=df['Contract'].unique(),
    default=df['Contract'].unique()
)

internet_filter = st.sidebar.multiselect(
    "Internet Service",
    options=df['Internet_Service'].unique(),
    default=df['Internet_Service'].unique()
)

senior_filter = st.sidebar.multiselect(
    "Senior Citizen",
    options=df['Senior_Citizen'].unique(),
    default=df['Senior_Citizen'].unique()
)


df_filtered = df[
    (df['Contract'].isin(contract_filter)) &
    (df['Internet_Service'].isin(internet_filter)) &
    (df['Senior_Citizen'].isin(senior_filter))
]


st.title("📊 Customer Churn Prediction Dashboard")
st.markdown("**Karmix Tech Internship | IBM Telco Dataset**")
st.markdown("---")


total     = len(df_filtered)
churned   = df_filtered['Churn_Value'].sum()
churn_rate = round((churned / total) * 100, 2)
revenue_at_risk = round(df_filtered[df_filtered['Churn_Value']==1]['Monthly_Charges'].sum(), 2)
avg_tenure = round(df_filtered['Tenure_Months'].mean(), 1)

col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("👥 Total Customers", f"{total:,}")
col2.metric("🚨 Churned", f"{churned:,}")
col3.metric("📉 Churn Rate", f"{churn_rate}%")
col4.metric("💸 Revenue at Risk", f"${revenue_at_risk:,}")
col5.metric("📅 Avg Tenure", f"{avg_tenure} months")

st.markdown("---")


col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("Overall Churn Rate")
    fig, ax = plt.subplots()
    counts = df_filtered['Churn_Label'].value_counts()
    ax.pie(counts, labels=counts.index, autopct='%1.1f%%',
           colors=['#2ecc71','#e74c3c'], startangle=90)
    st.pyplot(fig)
    plt.close()

with col2:
    st.subheader("Churn by Contract Type")
    fig, ax = plt.subplots()
    data = df_filtered.groupby('Contract')['Churn_Value'].mean() * 100
    data.plot(kind='bar', ax=ax, color=['#e74c3c','#f39c12','#2ecc71'], edgecolor='black')
    ax.set_ylabel("Churn Rate (%)")
    ax.tick_params(axis='x', rotation=15)
    st.pyplot(fig)
    plt.close()

with col3:
    st.subheader("Churn by Internet Service")
    fig, ax = plt.subplots()
    data = df_filtered.groupby('Internet_Service')['Churn_Value'].mean() * 100
    data.plot(kind='bar', ax=ax, color=['#3498db','#9b59b6','#1abc9c'], edgecolor='black')
    ax.set_ylabel("Churn Rate (%)")
    ax.tick_params(axis='x', rotation=15)
    st.pyplot(fig)
    plt.close()


col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("Tenure Distribution")
    fig, ax = plt.subplots()
    df_filtered[df_filtered['Churn_Label']=='No']['Tenure_Months'].hist(
        ax=ax, bins=30, alpha=0.6, color='#2ecc71', label='No Churn')
    df_filtered[df_filtered['Churn_Label']=='Yes']['Tenure_Months'].hist(
        ax=ax, bins=30, alpha=0.6, color='#e74c3c', label='Churned')
    ax.legend()
    ax.set_xlabel("Tenure (Months)")
    st.pyplot(fig)
    plt.close()

with col2:
    st.subheader("Churn by Payment Method")
    fig, ax = plt.subplots()
    data = df_filtered.groupby('Payment_Method')['Churn_Value'].mean() * 100
    data.sort_values().plot(kind='barh', ax=ax, color='#e67e22', edgecolor='black')
    ax.set_xlabel("Churn Rate (%)")
    st.pyplot(fig)
    plt.close()

with col3:
    st.subheader("Senior vs Non-Senior Churn")
    fig, ax = plt.subplots()
    data = df_filtered.groupby('Senior_Citizen')['Churn_Value'].mean() * 100
    data.index = ['Non-Senior','Senior']
    data.plot(kind='bar', ax=ax, color=['#27ae60','#e74c3c'], edgecolor='black')
    ax.set_ylabel("Churn Rate (%)")
    ax.tick_params(axis='x', rotation=0)
    st.pyplot(fig)
    plt.close()


col1, col2 = st.columns(2)

with col1:
    st.subheader("Top 10 Churn Reasons")
    fig, ax = plt.subplots(figsize=(8,5))
    reasons = df_filtered[df_filtered['Churn_Reason']!='Still Active']['Churn_Reason'].value_counts().head(10)
    reasons.sort_values().plot(kind='barh', ax=ax, color='#c0392b', edgecolor='black')
    ax.set_xlabel("Number of Customers")
    st.pyplot(fig)
    plt.close()

with col2:
    st.subheader("Top 10 Cities by Churn")
    fig, ax = plt.subplots(figsize=(8,5))
    city = df_filtered[df_filtered['Churn_Label']=='Yes']['City'].value_counts().head(10)
    city.sort_values().plot(kind='barh', ax=ax, color='#8e44ad', edgecolor='black')
    ax.set_xlabel("Churned Customers")
    st.pyplot(fig)
    plt.close()

st.markdown("---")


st.title("🤖 ML Model — Churn Prediction")

cols_ml = ['Gender','Senior_Citizen','Partner','Dependents','Tenure_Months',
           'Phone_Service','Multiple_Lines','Internet_Service','Online_Security',
           'Online_Backup','Device_Protection','Tech_Support','Streaming_TV',
           'Streaming_Movies','Contract','Paperless_Billing','Payment_Method',
           'Monthly_Charges','Total_Charges','Churn_Value']

df_ml = df[cols_ml].copy()
le = LabelEncoder()
for col in df_ml.select_dtypes(include='object').columns:
    df_ml[col] = le.fit_transform(df_ml[col].astype(str))

X = df_ml.drop('Churn_Value', axis=1)
y = df_ml['Churn_Value']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)
accuracy = accuracy_score(y_test, model.predict(X_test))

col1, col2 = st.columns(2)
col1.metric("✅ Model Accuracy", f"{accuracy*100:.2f}%")
col2.metric("🧪 Algorithm", "Logistic Regression")


st.subheader("Feature Importance")
fig, ax = plt.subplots(figsize=(10, 5))
importance = pd.Series(abs(model.coef_[0]), index=X.columns).sort_values(ascending=True)
importance.plot(kind='barh', ax=ax, color='#e74c3c', edgecolor='black')
ax.set_xlabel("Importance Score")
st.pyplot(fig)
plt.close()

st.markdown("---")


st.title("💡 Business Recommendations")
st.markdown("""
1. **Contract Upgrade Campaign** — Month-to-month customers ko 1-2 year contract pe migrate karo with discounts (42% → 3% churn)
2. **Fiber Optic Quality Improvement** — Fiber optic customers ki complaints address karo — speed aur reliability improve karo
3. **New Customer Retention** — Pehle 10 months mein new customers ko special offers do — ye period sabse high-risk hai
4. **Electronic Check Users** — In customers ko auto-pay pe migrate karo with incentives
5. **Senior Citizen Support** — Senior customers ke liye dedicated support aur simplified plans banao (41% churn rate)
6. **Competitor Response** — Top churn reason competitor hai — pricing aur download speed offers review karo
""")

st.markdown("---")
st.caption("Karmix Tech Internship | Customer Churn Prediction Dashboard | IBM Telco Dataset")



#  streamlit run D:\Karmix_Tech\Customer_churn_prediction\Dashboard\dashboard.py
