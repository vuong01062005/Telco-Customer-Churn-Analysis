import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, auc
import warnings

warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Telco Customer Churn Analysis",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main {
        padding: 0rem 1rem;
    }
    .stMetric {
        background-color: #f0f2f6;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        padding-left: 20px;
        padding-right: 20px;
        background-color: #f0f2f6;
        border-radius: 10px 10px 0 0;
    }
    .stTabs [aria-selected="true"] {
        background-color: #ff4b4b;
        color: white;
    }
    h1 {
        color: #1f77b4;
        font-weight: 700;
    }
    h2 {
        color: #ff4b4b;
        font-weight: 600;
        margin-top: 20px;
    }
    h3 {
        color: #2c3e50;
        font-weight: 500;
    }
    </style>
    """, unsafe_allow_html=True)

# Initialize session state
if 'data' not in st.session_state:
    st.session_state.data = None
if 'model' not in st.session_state:
    st.session_state.model = None
if 'scaler' not in st.session_state:
    st.session_state.scaler = None
if 'label_encoders' not in st.session_state:
    st.session_state.label_encoders = {}


# Data preprocessing function
def preprocess_data(df):
    df = df.copy()

    # Handle TotalCharges
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df['TotalCharges'].fillna(df['TotalCharges'].median(), inplace=True)

    # Drop customerID
    if 'customerID' in df.columns:
        df = df.drop('customerID', axis=1)

    return df


# Encode categorical features
def encode_features(df, fit=True):
    df = df.copy()
    categorical_cols = df.select_dtypes(include=['object']).columns

    if fit:
        st.session_state.label_encoders = {}
        for col in categorical_cols:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])
            st.session_state.label_encoders[col] = le
    else:
        for col in categorical_cols:
            if col in st.session_state.label_encoders:
                le = st.session_state.label_encoders[col]
                df[col] = le.transform(df[col])

    return df


# Header
st.markdown("""
    <div style='text-align: center; padding: 20px; background: linear-gradient(90deg, #1e3c72 0%, #2a5298 100%); border-radius: 10px; margin-bottom: 30px;'>
        <h1 style='color: white; margin: 0; font-size: 3em;'>📊 Telco Customer Churn Analysis</h1>
        <p style='color: #e0e0e0; font-size: 1.2em; margin-top: 10px;'>Phân Tích Dữ Liệu & Dự Đoán Khách Hàng Rời Bỏ</p>
    </div>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.image("https://img.icons8.com/clouds/200/000000/analytics.png", width=150)
    st.title("⚙️ Control Panel")

    st.markdown("---")

    # Upload section
    st.subheader("📁 Tải dữ liệu")
    uploaded_file = st.file_uploader("Tải lên file CSV", type=['csv'])

    if uploaded_file is not None:
        try:
            st.session_state.data = pd.read_csv(uploaded_file)
            st.success(f"✅ Đã tải {len(st.session_state.data)} dòng")
        except Exception as e:
            st.error(f"Error loading file: {e}")

    st.markdown("---")

    # Navigation
    if st.session_state.data is not None:
        page = st.radio(
            "📋 Bảng điều hướng",
            ["🏠 Trang chủ", "📈 EDA", "🤖 Mô hình ML", "🎯 Dự đoán"],
            label_visibility="collapsed"
        )
    else:
        st.warning("⚠️ Vui lòng tải lên file CSV trước")
        page = "🏠 Trang chủ"

    st.markdown("---")

    # Initialize filter variables
    gender_filter = None
    senior_filter = "All"
    partner_filter = None
    dependents_filter = None
    phone_filter = None
    internet_filter = None
    security_filter = None
    support_filter = None
    contract_filter = None
    payment_filter = None
    paperless_filter = None
    churn_filter = None

    # Filters (only show if data is loaded)
    if st.session_state.data is not None:
        st.subheader("🔍 Bộ lọc")

        # Demographics Filters
        with st.expander("👥 Nhân khẩu học", expanded=True):
            gender_options = st.session_state.data['gender'].unique().tolist()
            gender_filter = st.multiselect(
                "Giới tính",
                gender_options,
                default=gender_options
            )

            senior_filter = st.radio(
                "Người cao tuổi?",
                ["All", "Yes", "No"],
                horizontal=True
            )

            if 'Partner' in st.session_state.data.columns:
                partner_options = st.session_state.data['Partner'].unique().tolist()
                partner_filter = st.multiselect(
                    "Có gia đình",
                    partner_options,
                    default=partner_options
                )

            if 'Dependents' in st.session_state.data.columns:
                dependents_options = st.session_state.data['Dependents'].unique().tolist()
                dependents_filter = st.multiselect(
                    "Người phụ thuộc",
                    dependents_options,
                    default=dependents_options
                )

        # Service Filters
        with st.expander("📞 Dịch vụ", expanded=False):
            if 'PhoneService' in st.session_state.data.columns:
                phone_options = st.session_state.data['PhoneService'].unique().tolist()
                phone_filter = st.multiselect(
                    "Dịch vụ điện thoại",
                    phone_options,
                    default=phone_options
                )

            if 'InternetService' in st.session_state.data.columns:
                internet_options = st.session_state.data['InternetService'].unique().tolist()
                internet_filter = st.multiselect(
                    "Dịch vụ Internet",
                    internet_options,
                    default=internet_options
                )

            if 'OnlineSecurity' in st.session_state.data.columns:
                security_options = st.session_state.data['OnlineSecurity'].unique().tolist()
                security_filter = st.multiselect(
                    "Bảo mật trực tuyến",
                    security_options,
                    default=security_options
                )

            if 'TechSupport' in st.session_state.data.columns:
                support_options = st.session_state.data['TechSupport'].unique().tolist()
                support_filter = st.multiselect(
                    "Hỗ trợ kỹ thuật",
                    support_options,
                    default=support_options
                )

        # Contract & Payment Filters
        with st.expander("💼 Hợp đồng & Thanh toán", expanded=False):
            if 'Contract' in st.session_state.data.columns:
                contract_options = st.session_state.data['Contract'].unique().tolist()
                contract_filter = st.multiselect(
                    "Loại hợp đồng",
                    contract_options,
                    default=contract_options
                )

            if 'PaymentMethod' in st.session_state.data.columns:
                payment_options = st.session_state.data['PaymentMethod'].unique().tolist()
                payment_filter = st.multiselect(
                    "Phương thức thanh toán",
                    payment_options,
                    default=payment_options
                )

            if 'PaperlessBilling' in st.session_state.data.columns:
                paperless_options = st.session_state.data['PaperlessBilling'].unique().tolist()
                paperless_filter = st.multiselect(
                    "Hóa đơn điện tử",
                    paperless_options,
                    default=paperless_options
                )

        # Churn Filter
        with st.expander("🎯 Tình trạng", expanded=False):
            if 'Churn' in st.session_state.data.columns:
                churn_options = st.session_state.data['Churn'].unique().tolist()
                churn_filter = st.multiselect(
                    "Tình trạng rời bỏ",
                    churn_options,
                    default=churn_options
                )

    st.markdown("---")

# Main content based on page selection
if st.session_state.data is None:
    st.warning("⚠️ Please upload a CSV file to begin analysis")
    st.info("👆 Use the sidebar to upload your Telco Customer Churn dataset")

else:
    df = st.session_state.data.copy()

    # Apply ALL filters
    if gender_filter and len(gender_filter) > 0:
        df = df[df['gender'].isin(gender_filter)]

    if senior_filter != "All":
        senior_val = 1 if senior_filter == "Yes" else 0
        df = df[df['SeniorCitizen'] == senior_val]

    if partner_filter and len(partner_filter) > 0:
        df = df[df['Partner'].isin(partner_filter)]

    if dependents_filter and len(dependents_filter) > 0:
        df = df[df['Dependents'].isin(dependents_filter)]

    if phone_filter and len(phone_filter) > 0:
        df = df[df['PhoneService'].isin(phone_filter)]

    if internet_filter and len(internet_filter) > 0:
        df = df[df['InternetService'].isin(internet_filter)]

    if security_filter and len(security_filter) > 0:
        df = df[df['OnlineSecurity'].isin(security_filter)]

    if support_filter and len(support_filter) > 0:
        df = df[df['TechSupport'].isin(support_filter)]

    if contract_filter and len(contract_filter) > 0:
        df = df[df['Contract'].isin(contract_filter)]

    if payment_filter and len(payment_filter) > 0:
        df = df[df['PaymentMethod'].isin(payment_filter)]

    if paperless_filter and len(paperless_filter) > 0:
        df = df[df['PaperlessBilling'].isin(paperless_filter)]

    if churn_filter and len(churn_filter) > 0:
        df = df[df['Churn'].isin(churn_filter)]

    # Display filter info
    if len(df) < len(st.session_state.data):
        st.info(f"🔎 Đã lọc: {len(df):,} / {len(st.session_state.data):,} khách hàng")

    if page == "🏠 Trang chủ":
        # KPIs
        st.markdown("## 📊 Các chỉ số chính")

        col1, col2, col3, col4, col5 = st.columns(5)

        total_customers = len(df)
        churn_count = df['Churn'].value_counts().get('Yes', 0)
        churn_rate = (churn_count / total_customers * 100) if total_customers > 0 else 0
        avg_monthly = df['MonthlyCharges'].mean() if len(df) > 0 else 0
        df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
        total_revenue = df['TotalCharges'].sum() / 1000000 if len(df) > 0 else 0
        avg_tenure = df['tenure'].mean() if len(df) > 0 else 0

        with col1:
            st.metric(
                label="Tổng số khách hàng",
                value=f"{total_customers:,}"
            )

        with col2:
            st.metric(
                label="Tỷ lệ rời bỏ",
                value=f"{churn_rate:.1f}%"
            )

        with col3:
            st.metric(
                label="Cước trung bình hàng tháng",
                value=f"${avg_monthly:.2f}"
            )

        with col4:
            st.metric(
                label="Tổng doanh thu",
                value=f"${total_revenue:.2f}M"
            )

        with col5:
            st.metric(
                label="Thời gian sử dụng trung bình",
                value=f"{avg_tenure:.0f} months"
            )

        st.markdown("---")

        # Charts Row 1
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("### 📉 Phân tích khách hàng rời bỏ")
            churn_counts = df['Churn'].value_counts()
            fig = go.Figure(data=[go.Pie(
                labels=churn_counts.index,
                values=churn_counts.values,
                hole=.4,
                marker_colors=['#2ecc71', '#e74c3c']
            )])
            fig.update_layout(height=350)
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.markdown("### 💰 Cước trung bình theo loại hợp đồng")
            contract_charges = df.groupby('Contract')['MonthlyCharges'].mean().sort_values(ascending=False)
            fig = go.Figure(data=[
                go.Bar(
                    x=contract_charges.index,
                    y=contract_charges.values,
                    marker_color=['#3498db', '#9b59b6', '#e67e22']
                )
            ])
            fig.update_layout(height=350, showlegend=False, yaxis_title="Cước trung bình hàng tháng ($)")
            st.plotly_chart(fig, use_container_width=True)

        st.markdown("---")

        # Charts Row 2
        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("### 👥 Phân tích thời gian sử dụng dịch vụ")
            fig = go.Figure(data=[go.Histogram(
                x=df['tenure'],
                nbinsx=30,
                marker_color='#1abc9c'
            )])
            fig.update_layout(height=300, showlegend=False, xaxis_title="Thời gian sử dụng (tháng)",
                              yaxis_title="Số lượng")
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.markdown("### 🌐 Loại dịch vụ Internet")
            internet_counts = df['InternetService'].value_counts()
            fig = go.Figure(data=[go.Pie(
                labels=internet_counts.index,
                values=internet_counts.values,
                marker_colors=['#e74c3c', '#3498db', '#95a5a6']
            )])
            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True)

        with col3:
            st.markdown("### 💳 Phương Thức Thanh Toán")
            payment_counts = df['PaymentMethod'].value_counts()
            fig = go.Figure(data=[go.Bar(
                y=payment_counts.index,
                x=payment_counts.values,
                orientation='h',
                marker_color='#9b59b6'
            )])
            fig.update_layout(height=300, showlegend=False, xaxis_title="Count")
            st.plotly_chart(fig, use_container_width=True)

        st.markdown("---")

        # Detailed Table
        st.markdown("### 📋 Tổng quan dữ liệu khách hàng")
        st.dataframe(df.head(100), use_container_width=True, height=400)

    elif page == "📈 EDA":
        st.markdown("## 📊 Phân Tích Dữ Liệu")

        tab1, tab2, tab3, tab4 = st.tabs(["👤 Nhân khẩu học", "📞 Dịch vụ", "💰 Tài chính", "🔗 Tương quan"])

        with tab1:
            col1, col2 = st.columns(2)

            with col1:
                st.markdown("### Phân tích Giới Tính")
                gender_churn = pd.crosstab(df['gender'], df['Churn'])
                fig = go.Figure()
                fig.add_trace(
                    go.Bar(name='Không rời bỏ', x=gender_churn.index, y=gender_churn['No'], marker_color='#2ecc71'))
                fig.add_trace(
                    go.Bar(name='Rời bỏ', x=gender_churn.index, y=gender_churn['Yes'], marker_color='#e74c3c'))
                fig.update_layout(barmode='group', height=400, yaxis_title="Số lượng")
                st.plotly_chart(fig, use_container_width=True)

            with col2:
                st.markdown("### Phân tích khách hàng cao tuổi")
                senior_churn = pd.crosstab(df['SeniorCitizen'], df['Churn'])
                senior_labels = ['Không cao tuổi', 'Cao tuổi']
                fig = go.Figure()
                fig.add_trace(
                    go.Bar(name='Không rời bỏ', x=senior_labels, y=senior_churn['No'].values, marker_color='#3498db'))
                fig.add_trace(
                    go.Bar(name='Rời bỏ', x=senior_labels, y=senior_churn['Yes'].values, marker_color='#e67e22'))
                fig.update_layout(barmode='group', height=400, yaxis_title="Số lượng")
                st.plotly_chart(fig, use_container_width=True)

            st.markdown("### Tác Động của Người có gia đình & Người Phụ Thuộc")
            col1, col2 = st.columns(2)

            with col1:
                partner_churn = pd.crosstab(df['Partner'], df['Churn'], normalize='index') * 100
                fig = go.Figure()
                fig.add_trace(
                    go.Bar(name='Không rời bỏ', x=partner_churn.index, y=partner_churn['No'], marker_color='#2ecc71'))
                fig.add_trace(
                    go.Bar(name='Rời bỏ', x=partner_churn.index, y=partner_churn['Yes'], marker_color='#e74c3c'))
                fig.update_layout(barmode='stack', height=300, title="Người có gia đình", yaxis_title="Phần trăm (%)")
                st.plotly_chart(fig, use_container_width=True)

            with col2:
                dep_churn = pd.crosstab(df['Dependents'], df['Churn'], normalize='index') * 100
                fig = go.Figure()
                fig.add_trace(go.Bar(name='Không rời bỏ', x=dep_churn.index, y=dep_churn['No'], marker_color='#2ecc71'))
                fig.add_trace(go.Bar(name='Rời bỏ', x=dep_churn.index, y=dep_churn['Yes'], marker_color='#e74c3c'))
                fig.update_layout(barmode='stack', height=300, title="Người phụ thuộc", yaxis_title="Phần trăm (%)")
                st.plotly_chart(fig, use_container_width=True)

        with tab2:
            st.markdown("### Phân Tích Đăng Ký Dịch Vụ")

            service_cols = ['PhoneService', 'MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup',
                            'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies']

            col1, col2 = st.columns(2)

            with col1:
                st.markdown("#### Số Lượng Đăng Ký Dịch Vụ")
                service_counts = {}
                for col in service_cols:
                    if col in df.columns:
                        if col == 'InternetService':
                            service_counts[col] = (df[col] != 'No').sum()
                        else:
                            service_counts[col] = df[col].value_counts().get('Yes', 0)

                fig = go.Figure(data=[go.Bar(
                    x=list(service_counts.keys()),
                    y=list(service_counts.values()),
                    marker_color='#1abc9c'
                )])
                fig.update_layout(
                    xaxis_tickangle=-45,
                    height=400,
                    yaxis_title="Số lượng khách hàng"
                )
                st.plotly_chart(fig, use_container_width=True)

            with col2:
                st.markdown("#### Tỷ lệ rời bỏ theo loại Internet")
                if 'InternetService' in df.columns:
                    internet_churn = pd.crosstab(df['InternetService'], df['Churn'], normalize='index') * 100
                    fig = go.Figure()
                    fig.add_trace(
                        go.Bar(name='Không rời bỏ', x=internet_churn.index, y=internet_churn['No'],
                               marker_color='#2ecc71'))
                    fig.add_trace(
                        go.Bar(name='Rời bỏ', x=internet_churn.index, y=internet_churn['Yes'], marker_color='#e74c3c'))
                    fig.update_layout(barmode='stack', height=400, yaxis_title="Phần trăm (%)")
                    st.plotly_chart(fig, use_container_width=True)

            st.markdown("#### Lượng đăng ký Dịch Vụ Bổ Sung")

            additional_services = [
                'OnlineSecurity',
                'OnlineBackup',
                'DeviceProtection',
                'TechSupport',
                'StreamingTV',
                'StreamingMovies'
            ]

            fig = make_subplots(
                rows=3, cols=2,
                specs=[[{'type': 'domain'}, {'type': 'domain'}],
                    [{'type': 'domain'}, {'type': 'domain'}],
                    [{'type': 'domain'}, {'type': 'domain'}]],
                subplot_titles=additional_services
            )

            for idx, service in enumerate(additional_services):
                if service in df.columns:
                    row = idx // 2 + 1
                    col = idx % 2 + 1

                    data = df[service].value_counts(normalize=True) * 100

                    fig.add_trace(
                        go.Pie(
                            labels=data.index,
                            values=data.values,
                            name=service,
                            hole=0.4,
                            textinfo='percent+label'
                        ),
                        row=row, col=col
                    )

            fig.update_layout(
                height=800,
                showlegend=True
            )

            st.plotly_chart(fig, use_container_width=True)

        with tab3:
            st.markdown("### Phân tích tài chính")

            col1, col2 = st.columns(2)

            with col1:
                st.markdown("#### Cước hàng tháng theo tình trạng rời bỏ")
                fig = go.Figure()
                for churn_val in df['Churn'].unique():
                    fig.add_trace(go.Box(
                        y=df[df['Churn'] == churn_val]['MonthlyCharges'],
                        name=churn_val,
                        marker_color='#2ecc71' if churn_val == 'No' else '#e74c3c'
                    ))
                fig.update_layout(height=400, yaxis_title="Cước hàng tháng ($)")
                st.plotly_chart(fig, use_container_width=True)

            with col2:
                st.markdown("#### Tổng Chi Phí vs Thời Gian Sử Dụng")
                df_clean = df.dropna(subset=['TotalCharges', 'tenure'])
                fig = px.scatter(
                    df_clean,
                    x='tenure',
                    y='TotalCharges',
                    color='Churn',
                    color_discrete_map={'No': '#2ecc71', 'Yes': '#e74c3c'},
                    opacity=0.6
                )
                fig.update_layout(height=400, xaxis_title="Thời gian sử dụng (tháng)", yaxis_title="Tổng chi phí ($)")
                st.plotly_chart(fig, use_container_width=True)

            st.markdown("#### Phân Tích Loại Hợp Đồng")
            col1, col2 = st.columns(2)

            with col1:
                contract_churn = pd.crosstab(df['Contract'], df['Churn'], normalize='index') * 100
                fig = go.Figure()
                fig.add_trace(
                    go.Bar(name='Không rời bỏ', x=contract_churn.index, y=contract_churn['No'], marker_color='#2ecc71'))
                fig.add_trace(
                    go.Bar(name='Rời bỏ', x=contract_churn.index, y=contract_churn['Yes'], marker_color='#e74c3c'))
                fig.update_layout(barmode='stack', height=400, title="Tỷ lệ rời bỏ theo hợp đồng",
                                  yaxis_title="Phần trăm (%)")
                st.plotly_chart(fig, use_container_width=True)

            with col2:
                payment_churn = pd.crosstab(df['PaymentMethod'], df['Churn'], normalize='index') * 100
                fig = go.Figure()
                fig.add_trace(go.Bar(name='Không rời bỏ', y=payment_churn.index, x=payment_churn['No'], orientation='h',
                                     marker_color='#2ecc71'))
                fig.add_trace(go.Bar(name='Rời bỏ', y=payment_churn.index, x=payment_churn['Yes'], orientation='h',
                                     marker_color='#e74c3c'))
                fig.update_layout(barmode='stack', height=400, title="Tỉ lệ rời bỏ theo phương thức thanh toán",
                                  xaxis_title="Phần trăm (%)")
                st.plotly_chart(fig, use_container_width=True)

            st.markdown("### PaperlessBilling vs PaymentMethod")
            crosstab = pd.crosstab(df['PaymentMethod'], df['PaperlessBilling'], normalize='index') * 100
            crosstab = crosstab.round(1)

            fig = go.Figure(data=go.Heatmap(
                z=crosstab.values,
                x=crosstab.columns,
                y=crosstab.index,
                colorscale='Blues',
                text=crosstab.values,
                texttemplate="%{text}%",
                textfont={"size": 16},
                colorbar=dict(title="%", thickness=20)
            ))

            fig.update_layout(
                height=500,
                xaxis_title="Paperless Billing",
                yaxis_title="Payment Method",
                xaxis=dict(side="bottom")
            )
            st.plotly_chart(fig, use_container_width=True)

            st.markdown("#### Chi phí")
            col1, col2 = st.columns(2)

            with col1:
                st.markdown("###### Chi phí mỗi tháng")
                fig, ax = plt.subplots(figsize=(10, 6))
                sns.histplot(
                    data=df,
                    x='MonthlyCharges',
                    kde=True,
                    hue='Churn',
                    ax=ax
                )
                ax.set_title("Phân bố MonthlyCharges theo Churn")
                st.pyplot(fig)

            with col2:
                df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
                st.markdown("###### Tổng chi phí")
                fig, ax = plt.subplots(figsize=(10, 6))
                sns.histplot(
                    data=df,
                    x='TotalCharges',
                    kde=True,
                    hue='Churn',
                    ax=ax
                )
                ax.set_title("Phân bố TotalCharges theo Churn")
                st.pyplot(fig)

        with tab4:
            st.markdown("### Tương Quan Đặc Trưng")

            # Prepare data for correlation
            df_corr = preprocess_data(df)
            df_encoded = encode_features(df_corr, fit=True)

            # Calculate correlation matrix
            corr_matrix = df_encoded.corr()

            # Focus on correlations with Churn
            churn_corr = corr_matrix['Churn'].sort_values(ascending=False)

            col1, col2 = st.columns([1, 2])

            with col1:
                st.markdown("#### Tương Quan Cao Nhất")
                st.dataframe(
                    churn_corr.to_frame().style.background_gradient(cmap='RdYlGn_r'),
                    height=600
                )

            with col2:
                st.markdown("#### Correlation Heatmap")

                # Select top features
                top_features = churn_corr.abs().nlargest(10).index.tolist()
                corr_subset = corr_matrix.loc[top_features, top_features]

                fig = go.Figure(data=go.Heatmap(
                    z=corr_subset.values,
                    x=corr_subset.columns,
                    y=corr_subset.columns,
                    colorscale='RdBu',
                    zmid=0,
                    text=np.round(corr_subset.values, 2),
                    texttemplate='%{text}',
                    textfont={"size": 10}
                ))
                fig.update_layout(height=600)
                st.plotly_chart(fig, use_container_width=True)

    elif page == "🤖 Mô hình ML":
        st.markdown("## 🤖 Mô hình Machine Learning")

        col1, col2 = st.columns([1, 2])

        with col1:
            model_type = st.selectbox(
                "Chọn mô hình",
                ["Logistic Regression", "Random Forest", "K-Nearest Neighbors (KNN)",
                 "Gaussian Naive Bayes", "Decision Tree", "XGBoost"]
            )

            st.markdown("### Siêu Tham Số")

            # Model-specific hyperparameters
            if model_type == "Random Forest":
                n_estimators = st.slider("Number of Trees", 50, 500, 100, step=50)
                max_depth = st.slider("Max Depth", 5, 50, 10)
                min_samples_split = st.slider("Min Samples Split", 2, 20, 2)

            elif model_type == "K-Nearest Neighbors (KNN)":
                n_neighbors = st.slider("Number of Neighbors (K)", 1, 30, 5)
                weights = st.selectbox("Weight Function", ["uniform", "distance"])

            elif model_type == "Decision Tree":
                max_depth_dt = st.slider("Max Depth", 3, 30, 10)
                min_samples_split_dt = st.slider("Min Samples Split", 2, 20, 2)
                criterion = st.selectbox("Criterion", ["gini", "entropy"])

            elif model_type == "XGBoost":
                n_estimators_xgb = st.slider("Number of Estimators", 50, 500, 100, step=50)
                learning_rate = st.slider("Learning Rate", 0.01, 0.3, 0.1, step=0.01)
                max_depth_xgb = st.slider("Max Depth", 3, 10, 3)

            test_size = st.slider("Kích thước tập test (%)", 10, 40, 20) / 100
            random_state = st.number_input("Random State", value=42)

            st.markdown("---")

            if st.button("🚀 Huấn luyện mô hình", type="primary", use_container_width=True):
                with st.spinner("Đang huấn luyện mô hình..."):
                    try:
                        df_processed = preprocess_data(df)
                        df_encoded = encode_features(df_processed, fit=True)

                        X = df_encoded.drop('Churn', axis=1)
                        y = df_encoded['Churn']

                        X_train, X_test, y_train, y_test = train_test_split(
                            X, y, test_size=test_size, random_state=random_state, stratify=y
                        )

                        scaler = StandardScaler()
                        X_train_scaled = scaler.fit_transform(X_train)
                        X_test_scaled = scaler.transform(X_test)
                        st.session_state.scaler = scaler

                        # Model selection
                        if model_type == "Logistic Regression":
                            model = LogisticRegression(random_state=random_state, max_iter=1000)
                            X_train_use = X_train_scaled
                            X_test_use = X_test_scaled

                        elif model_type == "Random Forest":
                            model = RandomForestClassifier(
                                n_estimators=n_estimators,
                                max_depth=max_depth,
                                min_samples_split=min_samples_split,
                                random_state=random_state,
                                class_weight='balanced'
                            )
                            X_train_use = X_train
                            X_test_use = X_test

                        elif model_type == "K-Nearest Neighbors (KNN)":
                            model = KNeighborsClassifier(
                                n_neighbors=n_neighbors,
                                weights=weights
                            )
                            X_train_use = X_train_scaled
                            X_test_use = X_test_scaled

                        elif model_type == "Gaussian Naive Bayes":
                            model = GaussianNB()
                            X_train_use = X_train_scaled
                            X_test_use = X_test_scaled

                        elif model_type == "Decision Tree":
                            model = DecisionTreeClassifier(
                                max_depth=max_depth_dt,
                                min_samples_split=min_samples_split_dt,
                                criterion=criterion,
                                random_state=random_state
                            )
                            X_train_use = X_train
                            X_test_use = X_test

                        elif model_type == "XGBoost":
                            model = XGBClassifier(
                                n_estimators=n_estimators_xgb,
                                learning_rate=learning_rate,
                                max_depth=max_depth_xgb,
                                random_state=random_state,
                                eval_metric='logloss'
                            )
                            X_train_use = X_train
                            X_test_use = X_test

                        model.fit(X_train_use, y_train)
                        st.session_state.model = model
                        st.session_state.model_type = model_type
                        st.session_state.uses_scaling = model_type in ["Logistic Regression",
                                                                       "K-Nearest Neighbors (KNN)",
                                                                       "Gaussian Naive Bayes"]

                        y_pred = model.predict(X_test_use)
                        y_pred_proba = model.predict_proba(X_test_use)[:, 1]

                        st.session_state.metrics = {
                            'accuracy': accuracy_score(y_test, y_pred),
                            'precision': precision_score(y_test, y_pred),
                            'recall': recall_score(y_test, y_pred),
                            'f1': f1_score(y_test, y_pred),
                            'cm': confusion_matrix(y_test, y_pred),
                            'fpr': None,
                            'tpr': None,
                            'auc': None,
                            'X_test': X_test,
                            'y_test': y_test,
                            'y_pred_proba': y_pred_proba
                        }

                        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
                        roc_auc = auc(fpr, tpr)
                        st.session_state.metrics['fpr'] = fpr
                        st.session_state.metrics['tpr'] = tpr
                        st.session_state.metrics['auc'] = roc_auc

                        st.success("✅ Huấn luyện mô hình thành công!")
                        st.balloons()

                    except Exception as e:
                        st.error(f"Lỗi: {e}")

        with col2:
            st.markdown("### Hiệu suất mô hình")

            if 'metrics' in st.session_state:
                metrics = st.session_state.metrics

                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Độ chính xác tổng thể (Accuracy)", f"{metrics['accuracy'] * 100:.1f}%")
                with col2:
                    st.metric("Độ chính xác dương (Precision)", f"{metrics['precision'] * 100:.1f}%")
                with col3:
                    st.metric("Độ phủ (Recall)", f"{metrics['recall'] * 100:.1f}%")
                with col4:
                    st.metric("F1-Score", f"{metrics['f1'] * 100:.1f}%")

                st.markdown("#### Đường cong ROC")
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=metrics['fpr'],
                    y=metrics['tpr'],
                    mode='lines',
                    name=f'Mô hình (AUC={metrics["auc"]:.3f})',
                    line=dict(color='#e74c3c', width=3)
                ))
                fig.add_trace(go.Scatter(
                    x=[0, 1],
                    y=[0, 1],
                    mode='lines',
                    name='Ngẫu nhiên (AUC=0.50)',
                    line=dict(color='gray', width=2, dash='dash')
                ))
                fig.update_layout(
                    xaxis_title="False Positive Rate",
                    yaxis_title="True Positive Rate",
                    height=400
                )
                st.plotly_chart(fig, use_container_width=True)

                col1, col2 = st.columns(2)

                with col1:
                    st.markdown("#### Ma trận nhầm lẫn")
                    cm = metrics['cm']
                    fig = go.Figure(data=go.Heatmap(
                        z=cm,
                        x=['Dự đoán Không', 'Dự đoán Có'],
                        y=['Thực tế Không', 'Thực tế Có'],
                        colorscale='Blues',
                        text=cm,
                        texttemplate="%{text}",
                        textfont={"size": 20}
                    ))
                    fig.update_layout(height=300)
                    st.plotly_chart(fig, use_container_width=True)

                with col2:
                    st.markdown("#### Độ quan trọng của đặc trưng")
                    if hasattr(st.session_state.model, 'feature_importances_'):
                        importances = st.session_state.model.feature_importances_
                        feature_names = metrics['X_test'].columns
                        indices = np.argsort(importances)[-10:]

                        fig = go.Figure(data=[go.Bar(
                            y=[feature_names[i] for i in indices],
                            x=[importances[i] for i in indices],
                            orientation='h',
                            marker_color='#9b59b6'
                        )])
                        fig.update_layout(height=300, xaxis_title="Importance")
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        if hasattr(st.session_state.model, 'coef_'):
                            coef = np.abs(st.session_state.model.coef_[0])
                            feature_names = metrics['X_test'].columns
                            indices = np.argsort(coef)[-10:]

                            fig = go.Figure(data=[go.Bar(
                                y=[feature_names[i] for i in indices],
                                x=[coef[i] for i in indices],
                                orientation='h',
                                marker_color='#9b59b6'
                            )])
                            fig.update_layout(height=300, xaxis_title="Độ lớn hệ số")
                            st.plotly_chart(fig, use_container_width=True)
                        else:
                            st.info("Mô hình này không hỗ trợ feature importance")
            else:
                st.info("👆 Configure and train a model to see performance metrics")

    elif page == "🎯 Dự đoán":
        st.markdown("## 🎯 Dự đoán nguy cơ khách hàng rời bỏ")

        if st.session_state.model is None:
            st.warning("⚠️ Vui lòng huấn luyện mô hình trước trong mục Mô hình ML")
        else:
            col1, col2 = st.columns([2, 1])

            with col1:
                st.markdown("### 📝 Thông tin khách hàng")

                col_a, col_b = st.columns(2)

                with col_a:
                    gender = st.selectbox("Gender", ["Male", "Female"])
                    senior = st.selectbox("Senior Citizen", ["No", "Yes"])
                    partner = st.selectbox("Partner", ["No", "Yes"])
                    dependents = st.selectbox("Dependents", ["No", "Yes"])
                    tenure = st.number_input("Tenure (months)", min_value=0, max_value=72, value=12)
                    phone_service = st.selectbox("Phone Service", ["No", "Yes"])
                    multiple_lines = st.selectbox("Multiple Lines", ["No", "Yes", "No phone service"])

                with col_b:
                    internet = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
                    online_security = st.selectbox("Online Security", ["No", "Yes", "No internet service"])
                    online_backup = st.selectbox("Online Backup", ["No", "Yes", "No internet service"])
                    device_protection = st.selectbox("Device Protection", ["No", "Yes", "No internet service"])
                    tech_support = st.selectbox("Tech Support", ["No", "Yes", "No internet service"])
                    streaming_tv = st.selectbox("Streaming TV", ["No", "Yes", "No internet service"])
                    streaming_movies = st.selectbox("Streaming Movies", ["No", "Yes", "No internet service"])

                col_a, col_b = st.columns(2)

                with col_a:
                    contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
                    paperless = st.selectbox("Paperless Billing", ["No", "Yes"])

                with col_b:
                    payment = st.selectbox("Payment Method",
                                           ["Electronic check", "Mailed check", "Bank transfer (automatic)",
                                            "Credit card (automatic)"])
                    monthly_charges = st.number_input("Monthly Charges ($)", min_value=0.0, max_value=200.0, value=70.0)

                total_charges = tenure * monthly_charges

                st.markdown("---")

                if st.button("🔮 Dự đoán", type="primary", use_container_width=True):
                    try:
                        input_data = pd.DataFrame({
                            'gender': [gender],
                            'SeniorCitizen': [1 if senior == "Yes" else 0],
                            'Partner': [partner],
                            'Dependents': [dependents],
                            'tenure': [tenure],
                            'PhoneService': [phone_service],
                            'MultipleLines': [multiple_lines],
                            'InternetService': [internet],
                            'OnlineSecurity': [online_security],
                            'OnlineBackup': [online_backup],
                            'DeviceProtection': [device_protection],
                            'TechSupport': [tech_support],
                            'StreamingTV': [streaming_tv],
                            'StreamingMovies': [streaming_movies],
                            'Contract': [contract],
                            'PaperlessBilling': [paperless],
                            'PaymentMethod': [payment],
                            'MonthlyCharges': [monthly_charges],
                            'TotalCharges': [total_charges]
                        })

                        input_encoded = encode_features(input_data, fit=False)

                        if st.session_state.uses_scaling:
                            input_scaled = st.session_state.scaler.transform(input_encoded)
                        else:
                            input_scaled = input_encoded

                        prediction = st.session_state.model.predict(input_scaled)[0]
                        prediction_proba = st.session_state.model.predict_proba(input_scaled)[0]

                        churn_probability = prediction_proba[1] * 100

                        st.session_state.prediction_result = {
                            'prediction': prediction,
                            'probability': churn_probability
                        }

                        st.success("✅ Dự đoán thành công!")

                    except Exception as e:
                        st.error(f"Lỗi: {e}")

            with col2:
                st.markdown("### 📊 Kết quả dự đoán")

                if 'prediction_result' in st.session_state:
                    result = st.session_state.prediction_result
                    prob = result['probability']

                    if prob < 30:
                        risk_level = "Nguy cơ thấp"
                        risk_color = "#2ecc71"
                    elif prob < 60:
                        risk_level = "Nguy cơ trung bình"
                        risk_color = "#f39c12"
                    else:
                        risk_level = "Nguy cơ cao"
                        risk_color = "#e74c3c"

                    st.markdown(f"""
                        <div style='background: linear-gradient(135deg, {risk_color} 0%, {risk_color}dd 100%); 
                                    padding: 30px; border-radius: 15px; text-align: center; color: white;'>
                            <h2 style='color: white; margin: 0;'>Tỉ lệ rời bỏ</h2>
                            <h1 style='color: white; font-size: 4em; margin: 10px 0;'>{prob:.1f}%</h1>
                            <p style='font-size: 1.2em;'>{risk_level}</p>
                        </div>
                    """, unsafe_allow_html=True)

                    st.markdown("---")

                    st.markdown("### 🎯 Các yếu tố rủi ro chính")

                    risk_factors = []

                    if contract == "Month-to-month":
                        risk_factors.append(("Month-to-month contract", 85))
                    if monthly_charges > 70:
                        risk_factors.append(("High monthly charges", 75))
                    if tech_support == "No":
                        risk_factors.append(("No tech support", 60))
                    if payment == "Electronic check":
                        risk_factors.append(("Electronic check payment", 55))
                    if internet == "Fiber optic":
                        risk_factors.append(("Fiber optic internet", 50))
                    if tenure < 12:
                        risk_factors.append(("Short tenure", 65))

                    if risk_factors:
                        for factor, score in risk_factors[:5]:
                            st.markdown(f"**{factor}**")
                            st.progress(score / 100)
                    else:
                        st.info("No significant risk factors detected")

                    st.markdown("---")

                    if prob > 60:
                        st.warning(
                            "💡 Gợi ý: Cân nhắc tặng ưu đãi hợp đồng dài hạn và cải thiện dịch vụ hỗ trợ kỹ thuật")
                    elif prob > 30:
                        st.info("💡 Gợi ý: Theo dõi sát sao khách hàng và cung cấp ưu đãi giữ chân")
                    else:
                        st.success(
                            "💡 Tình trạng: Khách hàng có khả năng tiếp tục sử dụng dịch vụ. Hãy duy trì chất lượng tốt.")
                else:
                    st.info("👈 Nhập thông tin khách hàng và nhấn Dự đoán để xem kết quả")

# Footer
st.markdown("---")