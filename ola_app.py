import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score

# ---------------- Page Config ----------------
st.set_page_config(page_title="Ola Bike Ride Demand Forecast", layout="wide")

# ---------------- Background Styling ----------------
def set_background():
    sunset_url = "https://images.unsplash.com/photo-1501973801540-537f08ccae7b"
    st.markdown(
        f"""
        <style>
        .stApp {{
            background: url("{sunset_url}") no-repeat center center fixed;
            background-size: cover;
        }}
        h1, h2, h3, h4, h5, h6, p, .stMarkdown {{
            color: white !important;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.8);
        }}
        div[data-testid="stDataFrame"] {{
            background-color: rgba(255, 255, 255, 0.95) !important;
        }}
        .stButton>button {{
            background-color: #ff7f50;
            color: white;
            font-weight: bold;
            border-radius: 8px;
            border: none;
        }}
        .stButton>button:hover {{
            background-color: #ff5722;
            color: white;
        }}
        div[data-testid="stMetricValue"] {{
            color: white !important;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )
set_background()

# ---------------- Sidebar ----------------
st.sidebar.title("📊 Ola Bike Demand Forecast")
phase = st.sidebar.radio(
    "Select Phase",
    [
        "Phase 1: Data Cleaning & Transformation",
        "Phase 2: Clustering with K-Means",
        "Phase 3: Model Training & Evaluation",
        "Phase 4: Presentation & Discussion"
    ]
)

# ---------------- Load Dataset ----------------
@st.cache_data
def load_data():
    url = "https://drive.google.com/uc?id=1Qnpb1--5liNmMvxoe3iM6Rz4yJMkTDFH"
    try:
        data = pd.read_csv(url)
        return data
    except Exception as e:
        st.error(f"Error loading dataset: {e}")
        return pd.DataFrame()

df = load_data()

# ---------------- Phase 1 ----------------
if phase == "Phase 1: Data Cleaning & Transformation":
    st.title("🧹 Phase 1: Data Cleaning & Transformation")

    if df.empty:
        st.error("Dataset could not be loaded. Please check the URL or upload a CSV file.")
        uploaded_file = st.file_uploader("Upload CSV file", type=['csv'])
        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)
    
    if not df.empty:
        st.subheader("Raw Data Preview")
        st.dataframe(df.head(10))
        
        st.subheader("Dataset Info")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Rows", df.shape[0])
        with col2:
            st.metric("Total Columns", df.shape[1])
        with col3:
            st.metric("Missing Values", df.isnull().sum().sum())

        st.subheader("Missing Values by Column")
        missing_df = pd.DataFrame({
            'Column': df.columns,
            'Missing Count': df.isnull().sum().values,
            'Missing %': (df.isnull().sum().values / len(df) * 100).round(2)
        })
        st.dataframe(missing_df[missing_df['Missing Count'] > 0])

        # Fill missing values
        df_cleaned = df.copy()
        for col in df_cleaned.columns:
            if df_cleaned[col].dtype in [np.float64, np.int64]:
                df_cleaned[col] = df_cleaned[col].fillna(df_cleaned[col].median())
            else:
                if df_cleaned[col].isnull().sum() > 0:
                    mode_val = df_cleaned[col].mode()
                    if len(mode_val) > 0:
                        df_cleaned[col] = df_cleaned[col].fillna(mode_val[0])

        # Encode categorical variables
        df_encoded = df_cleaned.copy()
        encoders = {}
        categorical_cols = df_encoded.select_dtypes(include="object").columns
        
        if len(categorical_cols) > 0:
            st.subheader("Encoding Categorical Variables")
            for col in categorical_cols:
                le = LabelEncoder()
                df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))
                encoders[col] = le
                st.write(f"✅ Encoded column: **{col}**")

        st.subheader("Cleaned & Encoded Data")
        st.dataframe(df_encoded.head(10))

        # Outliers detection
        numeric_cols = df_encoded.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            target_col = st.selectbox("Select column to check for outliers:", numeric_cols)
            
            st.subheader(f"Outliers in {target_col}")
            Q1 = df_encoded[target_col].quantile(0.25)
            Q3 = df_encoded[target_col].quantile(0.75)
            IQR = Q3 - Q1
            outliers = df_encoded[
                (df_encoded[target_col] < (Q1 - 1.5 * IQR)) | 
                (df_encoded[target_col] > (Q3 + 1.5 * IQR))
            ]
            st.write(f"Found {len(outliers)} outliers")
            if len(outliers) > 0:
                st.dataframe(outliers.head(10))

            st.subheader(f"{target_col} Distribution")
            fig, ax = plt.subplots(figsize=(10, 5))
            sns.histplot(df_encoded[target_col], bins=30, kde=True, ax=ax, color='coral')
            ax.set_title(f"Distribution of {target_col}")
            ax.set_xlabel(target_col)
            ax.set_ylabel("Frequency")
            st.pyplot(fig)
            plt.close()

        # Save to session state
        st.session_state.cleaned_data = df_encoded
        st.session_state.encoders = encoders
        st.success("✅ Data cleaned, encoded, and saved for next phase!")

# ---------------- Phase 2 ----------------
elif phase == "Phase 2: Clustering with K-Means":
    st.title("🔀 Phase 2: Clustering with K-Means")

    if "cleaned_data" not in st.session_state:
        st.warning("⚠️ Please complete Phase 1 first.")
    else:
        df_cluster = st.session_state.cleaned_data.copy()

        # Select features for clustering
        all_cols = df_cluster.columns.tolist()
        st.subheader("Select Features for Clustering")
        
        default_features = [col for col in all_cols if col.lower() not in ['requests', 'demand', 'id', 'index']]
        selected_features = st.multiselect(
            "Choose features:",
            all_cols,
            default=default_features[:min(5, len(default_features))]
        )
        
        if len(selected_features) < 2:
            st.error("Please select at least 2 features for clustering.")
        else:
            X = df_cluster[selected_features]
            
            # Scale features
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)

            # Select number of clusters
            n_clusters = st.slider("Number of Clusters:", 2, 10, 4)

            # KMeans clustering
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            df_cluster["Cluster"] = kmeans.fit_predict(X_scaled)

            st.subheader("Clustered Data Preview")
            st.dataframe(df_cluster.head(10))
            
            # Cluster statistics
            st.subheader("Cluster Distribution")
            cluster_counts = df_cluster["Cluster"].value_counts().sort_index()
            
            fig, ax = plt.subplots(figsize=(10, 5))
            cluster_counts.plot(kind='bar', ax=ax, color='skyblue')
            ax.set_title("Data Points per Cluster")
            ax.set_xlabel("Cluster")
            ax.set_ylabel("Count")
            plt.xticks(rotation=0)
            st.pyplot(fig)
            plt.close()

            # Cluster visualization
            if len(selected_features) >= 2:
                st.subheader("Cluster Visualization")
                col1, col2 = st.columns(2)
                
                with col1:
                    x_axis = st.selectbox("X-axis:", selected_features, index=0)
                with col2:
                    y_axis = st.selectbox("Y-axis:", selected_features, index=min(1, len(selected_features)-1))
                
                fig, ax = plt.subplots(figsize=(10, 6))
                scatter = ax.scatter(
                    df_cluster[x_axis], 
                    df_cluster[y_axis],
                    c=df_cluster["Cluster"], 
                    cmap="viridis",
                    alpha=0.6,
                    s=50
                )
                ax.set_xlabel(x_axis)
                ax.set_ylabel(y_axis)
                ax.set_title(f"Clusters: {x_axis} vs {y_axis}")
                plt.colorbar(scatter, label="Cluster", ax=ax)
                st.pyplot(fig)
                plt.close()

            # Save to session state
            st.session_state.clustered_data = df_cluster
            st.session_state.scaler = scaler
            st.session_state.kmeans_model = kmeans
            st.success("✅ Clustering complete and saved for next phase!")

# ---------------- Phase 3 ----------------
elif phase == "Phase 3: Model Training & Evaluation":
    st.title("🤖 Phase 3: Model Training & Evaluation")

    if "clustered_data" not in st.session_state:
        st.warning("⚠️ Please complete Phase 2 first.")
    else:
        df_model = st.session_state.clustered_data.copy()

        # Select target variable
        st.subheader("Select Target Variable")
        numeric_cols = df_model.select_dtypes(include=[np.number]).columns.tolist()
        
        # Try to find common target names
        default_target = None
        for possible_name in ['Requests', 'Demand', 'requests', 'demand', 'target', 'Target']:
            if possible_name in numeric_cols:
                default_target = possible_name
                break
        
        if default_target is None and len(numeric_cols) > 0:
            default_target = numeric_cols[0]
        
        target_col = st.selectbox(
            "Choose target variable to predict:",
            numeric_cols,
            index=numeric_cols.index(default_target) if default_target in numeric_cols else 0
        )

        if target_col:
            X = df_model.drop(columns=[target_col])
            y = df_model[target_col]

            if X.shape[0] < 20:
                st.error("❌ Not enough rows for reliable training. Need at least 20 rows.")
            else:
                # Train-test split
                test_size = st.slider("Test set size (%):", 10, 40, 20) / 100
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=test_size, random_state=42
                )
                
                st.info(f"Training set: {len(X_train)} samples | Test set: {len(X_test)} samples")

                # Model selection
                st.subheader("Select Models to Train")
                model_options = {
                    "Linear Regression": LinearRegression(),
                    "Random Forest": RandomForestRegressor(random_state=42, n_estimators=100),
                    "Gradient Boosting": GradientBoostingRegressor(random_state=42, n_estimators=100)
                }
                
                selected_models = st.multiselect(
                    "Choose models:",
                    list(model_options.keys()),
                    default=list(model_options.keys())
                )

                # Add auto-train option
                auto_train = st.checkbox("Auto-train on page load", value=False)
                train_button = st.button("🚀 Train Models")
                
                if train_button or (auto_train and "best_model" not in st.session_state):
                    results = []
                    trained_models = {}
                    
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    for idx, name in enumerate(selected_models):
                        status_text.text(f"Training {name}...")
                        model = model_options[name]
                        
                        try:
                            # Cross-validation
                            cv_scores = cross_val_score(model, X_train, y_train, cv=min(5, len(X_train)//2), scoring="r2")
                            cv_mean = np.mean(cv_scores)
                            
                            # Train on full training set
                            model.fit(X_train, y_train)
                            
                            # Predictions
                            y_pred_train = model.predict(X_train)
                            y_pred_test = model.predict(X_test)
                            
                            # Metrics
                            train_r2 = r2_score(y_train, y_pred_train)
                            test_r2 = r2_score(y_test, y_pred_test)
                            train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
                            test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
                            
                            results.append({
                                "Model": name,
                                "CV R² Score": round(cv_mean, 4),
                                "Train R²": round(train_r2, 4),
                                "Test R²": round(test_r2, 4),
                                "Train RMSE": round(train_rmse, 2),
                                "Test RMSE": round(test_rmse, 2)
                            })
                            
                            trained_models[name] = {
                                "model": model,
                                "predictions": y_pred_test,
                                "test_r2": test_r2
                            }
                            
                        except Exception as e:
                            st.warning(f"⚠️ {name} failed: {str(e)}")
                        
                        progress_bar.progress((idx + 1) / len(selected_models))
                    
                    status_text.empty()
                    progress_bar.empty()

                    if results:
                        results_df = pd.DataFrame(results)
                        st.subheader("📊 Model Comparison")
                        st.dataframe(results_df)

                        # Find best model
                        best_idx = results_df["Test R²"].idxmax()
                        best_model_name = results_df.loc[best_idx, "Model"]
                        best_model_data = trained_models[best_model_name]
                        
                        st.success(f"🏆 Best Model: **{best_model_name}** (Test R² = {results_df.loc[best_idx, 'Test R²']})")

                        # Visualization
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.subheader("R² Score Comparison")
                            fig, ax = plt.subplots(figsize=(8, 5))
                            x = np.arange(len(results_df))
                            width = 0.35
                            ax.bar(x - width/2, results_df["Train R²"], width, label='Train R²', color='lightblue')
                            ax.bar(x + width/2, results_df["Test R²"], width, label='Test R²', color='coral')
                            ax.set_xlabel('Model')
                            ax.set_ylabel('R² Score')
                            ax.set_title('Model Performance Comparison')
                            ax.set_xticks(x)
                            ax.set_xticklabels(results_df["Model"], rotation=15, ha='right')
                            ax.legend()
                            ax.grid(axis='y', alpha=0.3)
                            st.pyplot(fig)
                            plt.close()
                        
                        with col2:
                            st.subheader("Actual vs Predicted")
                            fig, ax = plt.subplots(figsize=(8, 5))
                            ax.scatter(y_test, best_model_data["predictions"], alpha=0.6, color='purple')
                            ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "r--", lw=2)
                            ax.set_xlabel("Actual Values")
                            ax.set_ylabel("Predicted Values")
                            ax.set_title(f"{best_model_name}: Predictions")
                            ax.grid(alpha=0.3)
                            st.pyplot(fig)
                            plt.close()

                        # Save best model
                        st.session_state.best_model = best_model_name
                        st.session_state.best_model_object = best_model_data["model"]
                        st.session_state.model_results = results_df
                        st.session_state.target_column = target_col
                        
                        # Download model
                        model_filename = "best_model.pkl"
                        model_bytes = pickle.dumps(best_model_data["model"])
                        st.download_button(
                            "📥 Download Best Model",
                            data=model_bytes,
                            file_name=model_filename,
                            mime="application/octet-stream"
                        )
                    else:
                        st.error("❌ All models failed during training.")

# ---------------- Phase 4 ----------------
elif phase == "Phase 4: Presentation & Discussion":
    st.title("📢 Phase 4: Presentation & Discussion")

    if "best_model" not in st.session_state:
        st.warning("⚠️ Please complete Phase 3 first to generate the best model.")
        st.info("""
        **To complete Phase 3:**
        1. Go to **Phase 3: Model Training & Evaluation**
        2. Select your target variable (the column you want to predict)
        3. Choose which models you want to train
        4. Click the **🚀 Train Models** button
        5. Wait for training to complete
        6. Then return to this phase
        """)
        
        # Show workflow diagram
        st.subheader("📋 Complete Workflow")
        st.markdown("""
        ```
        Phase 1: Data Cleaning ✓
              ↓
        Phase 2: Clustering ✓
              ↓
        Phase 3: Model Training ← YOU ARE HERE (incomplete)
              ↓
        Phase 4: Presentation
        ```
        """)
    else:
        st.subheader("🎯 Project Summary")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            if "cleaned_data" in st.session_state:
                st.metric("Total Records", st.session_state.cleaned_data.shape[0])
        with col2:
            if "clustered_data" in st.session_state:
                st.metric("Clusters Created", st.session_state.clustered_data["Cluster"].nunique())
        with col3:
            if "model_results" in st.session_state:
                best_r2 = st.session_state.model_results["Test R²"].max()
                st.metric("Best R² Score", f"{best_r2:.3f}")

        st.subheader("📌 Key Factors Affecting Ride Demand")
        st.markdown("""
        - **Time of Day** ⏰: Peak hours (morning & evening rush) show significantly higher demand
        - **Day of Week** 📅: Weekdays vs weekends exhibit different demand patterns
        - **Weather Conditions** 🌦️: Rain and extreme temperatures negatively impact ride requests
        - **Temperature** 🌡️: Moderate, comfortable temperatures correlate with increased rides
        - **Geographic Clusters** 📍: High-activity urban zones consistently generate more requests
        - **Special Events** 🎉: Concerts, sports events, and holidays create demand spikes
        """)

        st.subheader("🏆 Best Model Performance")
        st.success(f"**Selected Model:** {st.session_state.best_model}")
        
        if "model_results" in st.session_state:
            st.dataframe(st.session_state.model_results)

        st.subheader("💡 Business Recommendations")
        st.markdown("""
        1. **Dynamic Pricing**: Implement surge pricing during peak demand periods
        2. **Driver Allocation**: Position more drivers in high-cluster zones during rush hours
        3. **Weather Adjustments**: Offer incentives during bad weather to ensure service availability
        4. **Predictive Scheduling**: Use forecasts to optimize driver shifts and reduce wait times
        5. **Marketing Campaigns**: Target low-demand periods with promotional offers
        """)

        st.subheader("⚠️ Model Limitations & Future Improvements")
        st.markdown("""
        **Current Limitations:**
        - Dataset may not capture full seasonal variations or long-term trends
        - Special events, holidays, and unexpected incidents are not modeled
        - External factors (traffic, economic conditions, competitor actions) are not included
        - Model assumes historical patterns will continue unchanged
        
        **Future Enhancements:**
        - Incorporate real-time traffic and event data via APIs
        - Add external economic indicators and competitor pricing
        - Implement time-series models (LSTM, ARIMA) for better temporal patterns
        - Advanced hyperparameter tuning with GridSearchCV or Bayesian optimization
        - Ensemble methods combining multiple model types
        - A/B testing framework to validate predictions in production
        """)
        
        st.subheader("📊 Export Final Report")
        if st.button("Generate Summary Report"):
            report = f"""
            OLA BIKE DEMAND FORECAST - FINAL REPORT
            ========================================
            
            Project Date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}
            
            DATASET OVERVIEW:
            - Total Records: {st.session_state.cleaned_data.shape[0] if 'cleaned_data' in st.session_state else 'N/A'}
            - Total Features: {st.session_state.cleaned_data.shape[1] if 'cleaned_data' in st.session_state else 'N/A'}
            - Target Variable: {st.session_state.get('target_column', 'N/A')}
            
            CLUSTERING RESULTS:
            - Number of Clusters: {st.session_state.clustered_data["Cluster"].nunique() if 'clustered_data' in st.session_state else 'N/A'}
            
            MODEL PERFORMANCE:
            - Best Model: {st.session_state.best_model}
            - Best Test R² Score: {st.session_state.model_results["Test R²"].max() if 'model_results' in st.session_state else 'N/A'}
            
            CONCLUSION:
            The {st.session_state.best_model} model demonstrates the best predictive performance
            for forecasting bike ride demand, enabling data-driven decision making for operations.
            """
            
            st.download_button(
                "📄 Download Report",
                data=report,
                file_name="ola_demand_forecast_report.txt",
                mime="text/plain"
            )