import streamlit as st
import pandas as pd

# We will import the ML/EDA libraries inside functions 
# to keep app startup fast and avoid errors if not installed.

import traceback
import os
import numpy as np

# Global Gemini Monkey-patch to prevent recursion loop during Streamlit reruns
try:
    from pandasai.llm import GoogleGemini
    if not hasattr(GoogleGemini, "_is_patched"):
        original_configure = GoogleGemini._configure
        def new_configure(self, api_key):
            # Using Gemini 2.5 Flash model (available in this environment)
            self.model = "gemini-2.5-flash"
            original_configure(self, api_key)
        
        GoogleGemini._configure = new_configure
        GoogleGemini._is_patched = True
except Exception:
    pass

@st.cache_data
def get_profile_html(df: pd.DataFrame):
    from ydata_profiling import ProfileReport
    minimal = len(df) > 10000
    profile = ProfileReport(df, explorative=not minimal, minimal=minimal, title="Data Profiling Report")
    return profile.to_html()

def generate_eda_report(df: pd.DataFrame):
    """Generates and displays Auto EDA report using ydata-profiling."""
    try:
        import streamlit.components.v1 as components
        
        with st.spinner("Generating EDA report... this may take a moment depending on dataset size."):
            html_content = get_profile_html(df)
            
        components.html(html_content, height=1000, scrolling=True)
    except ImportError:
        st.error("ydata-profiling is not installed.")
    except Exception as e:
        st.error(f"Error generating EDA report: {e}")

@st.cache_data
def cached_process_data(df: pd.DataFrame, drop_cols: tuple, num_impute: str, cat_impute: str, drop_nan_rows: bool, nan_threshold: int, remove_outliers: bool, normalize: bool, norm_method: str):
    original_shape = df.shape
    original_nans = df.isnull().sum().sum()
    proc_df = df.copy()
    
    if drop_cols:
        proc_df = proc_df.drop(columns=list(drop_cols))
        
    if num_impute != "None":
        numeric_cols = proc_df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if proc_df[col].isnull().any():
                if num_impute == "mean": val = proc_df[col].mean()
                elif num_impute == "median": val = proc_df[col].median()
                elif num_impute == "mode":
                    m = proc_df[col].mode()
                    val = m[0] if not m.empty else 0
                else:
                    val = 0
                proc_df[col] = proc_df[col].fillna(val)

    if cat_impute != "None":
        cat_cols = proc_df.select_dtypes(exclude=[np.number]).columns
        for col in cat_cols:
            if proc_df[col].isnull().any():
                if cat_impute == "mode":
                    m = proc_df[col].mode()
                    val = m[0] if not m.empty else "Missing"
                else: val = "Missing"
                proc_df[col] = proc_df[col].fillna(val)

    rows_to_drop = proc_df.isnull().any(axis=1).sum()
    loss_percentage = (rows_to_drop / original_shape[0]) * 100
    
    drop_allowed = False
    if drop_nan_rows:
        if loss_percentage <= nan_threshold:
            proc_df = proc_df.dropna()
            drop_allowed = True
            
    outliers_removed = 0
    if remove_outliers:
        numeric_cols = proc_df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            z_scores = np.abs((proc_df[col] - proc_df[col].mean()) / (proc_df[col].std() + 1e-9))
            mask = z_scores < 3
            outliers_removed += (len(proc_df) - mask.sum())
            proc_df = proc_df[mask]
            
    if normalize:
        numeric_cols = proc_df.select_dtypes(include=[np.number]).columns
        from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
        if norm_method == "zscore": scaler = StandardScaler()
        elif norm_method == "minmax": scaler = MinMaxScaler()
        else: scaler = RobustScaler()
        
        if not proc_df[numeric_cols].empty:
            proc_df[numeric_cols] = scaler.fit_transform(proc_df[numeric_cols])
            
    return proc_df, drop_allowed, loss_percentage, original_shape, original_nans

def data_prep_interface(df: pd.DataFrame):
    """Interface for cleaning and preparing data with live feedback."""
    st.markdown("### 🧹 Data cleaning & Preparation")
    
    # Initialize prep config in session state if not present
    if 'prep_config' not in st.session_state:
        st.session_state.prep_config = {
            'drop_cols': [],
            'numeric_imputation': 'mean',
            'remove_outliers': False,
            'normalize': False,
            'normalize_method': 'zscore',
            'fix_imbalance': False,
            'feature_selection': False
        }

    # Sidebar-like control for options
    col_settings, col_report = st.columns([1, 1.5])
    
    with col_settings:
        st.subheader("1. Configuration")
        
        # Columns to Drop
        drop_cols = st.multiselect("Select columns to drop", options=df.columns, default=st.session_state.prep_config['drop_cols'])
        st.session_state.prep_config['drop_cols'] = drop_cols
        
        # Imputation
        num_impute = st.selectbox("Numerical Imputation (Mean/Median/Zero)", ["None", "mean", "median", "zero"], index=1)
        st.session_state.prep_config['numeric_imputation'] = num_impute
        
        cat_impute = st.selectbox("Categorical Imputation (Mode)", ["None", "mode"], index=1)
        st.session_state.prep_config['cat_impute'] = cat_impute
              # Row Deletion
        st.markdown("---")
        st.subheader("2. Row Deletion (Smart Drop)")
        drop_nan_rows = st.checkbox("Drop rows with missing values", value=st.session_state.prep_config.get('drop_nan_rows', False))
        st.session_state.prep_config['drop_nan_rows'] = drop_nan_rows
        
        nan_threshold = st.slider("Max Row Loss % (Threshold)", 1, 100, 20, help="Only drop rows if we lose less than this % of total data.")
        st.session_state.prep_config['nan_threshold'] = nan_threshold

    # Sidebar-like control for options (Part 2)
    with col_settings:
        st.markdown("---")
        st.subheader("3. Feature Scaling & Outliers")
        
        # Outliers
        remove_outliers = st.checkbox("Remove Outliers (Z-score > 3)", value=st.session_state.prep_config['remove_outliers'])
        st.session_state.prep_config['remove_outliers'] = remove_outliers
        
        # Scaling
        normalize = st.checkbox("Normalize Features (Scaling)", value=st.session_state.prep_config['normalize'])
        st.session_state.prep_config['normalize'] = normalize
        
        norm_method = st.selectbox("Normalization Method", ["zscore", "minmax", "robust"], index=0, disabled=not normalize)
        st.session_state.prep_config['normalize_method'] = norm_method

        st.markdown("---")
        # Action Buttons
        c1, c2 = st.columns(2)
        apply_btn = c1.button("Apply & Save to Dataset", type="primary", use_container_width=True)
        
        # Reset Button (Visible if raw_df exists)
        if 'raw_df' in st.session_state:
            if c2.button("Reset to Original Data", use_container_width=True):
                st.session_state.df = st.session_state.raw_df.copy()
                st.session_state.prep_config = {'drop_cols': [], 'numeric_imputation': 'mean', 'remove_outliers': False, 'normalize': False}
                st.success("Data reset to original!")
                st.rerun()

    # Call the cached function instead of running synchronously
    proc_df, drop_allowed, loss_percentage, original_shape, original_nans = cached_process_data(
        df,
        drop_cols=tuple(st.session_state.prep_config.get('drop_cols', [])),
        num_impute=st.session_state.prep_config.get('numeric_imputation', 'mean'),
        cat_impute=st.session_state.prep_config.get('cat_impute', 'None'),
        drop_nan_rows=st.session_state.prep_config.get('drop_nan_rows', False),
        nan_threshold=st.session_state.prep_config.get('nan_threshold', 20),
        remove_outliers=st.session_state.prep_config.get('remove_outliers', False),
        normalize=st.session_state.prep_config.get('normalize', False),
        norm_method=st.session_state.prep_config.get('normalize_method', 'zscore')
    )


    with col_report:
        st.markdown('<div class="result-card">', unsafe_allow_html=True)
        st.subheader("📊 Transformation Report")
        
        # Show Metrics
        m1, m2, m3 = st.columns(3)
        m1.metric("Rows Removed", f"{original_shape[0] - proc_df.shape[0]}")
        m2.metric("NaNs Fixed", f"{original_nans - proc_df.isnull().sum().sum()}")
        m3.metric("Cols Dropped", f"{len(drop_cols)}")
        
        if drop_nan_rows and not drop_allowed:
            st.warning(f"⚠️ **Drop Blocked**: You would lose {loss_percentage:.1f}% of your data (>{nan_threshold}%). Please use Imputation instead or increase your threshold.")
        
        st.markdown("#### Cleaned vs. Original Stats")
        
        # Comparison Table
        stats_raw = df.describe().T[['mean', 'std', 'min', 'max']].head(5)
        stats_new = proc_df.describe().T[['mean', 'std', 'min', 'max']].head(5)
        
        tabs = st.tabs(["Processed Data Preview", "Statistical Comparison"])
        
        with tabs[0]:
            st.dataframe(proc_df.head(50), use_container_width=True)
            
        with tabs[1]:
            st.write("Original Stats (Top 5 Cols):")
            st.dataframe(stats_raw)
            st.write("Processed Stats (Top 5 Cols):")
            st.dataframe(stats_new)
        st.markdown('</div>', unsafe_allow_html=True)

    # Actual Application
    if apply_btn:
        st.session_state.df = proc_df
        st.success("Changes permanently applied to your dataset!")
        st.balloons()
        
    return st.session_state.df # Always return current state

def run_auto_ml(df: pd.DataFrame):
    """Runs Auto ML experiments using pycaret with advanced options."""
    st.markdown("### 🚀 Advanced Auto ML")
    
    if df is None:
        st.warning("Please prepare your data first.")
        return

    target_col = st.selectbox("Select Target Column", options=df.columns)
    task_type = st.radio("Select ML Task", ["Classification", "Regression"])
    
    # Get prep config from session state or use defaults
    prep_config = st.session_state.get('prep_config', {})
    
    # Use tabs for the 12-step workflow organization
    tab1, tab2, tab3, tab4 = st.tabs(["Step 4: Training", "Step 5-7: Optimization", "Step 8: Analysis", "Step 9: Finalize"])
    
    with tab1:
        st.subheader("Compare Baseline Models")
        if st.button("Run Auto ML Experiment", key="run_automl_btn"):
            st.info(f"Starting {task_type} experiment...")
            try:
                if task_type == "Classification":
                    from pycaret.classification import setup, compare_models, pull
                else:
                    from pycaret.regression import setup, compare_models, pull
                
                with st.spinner("Setting up experiment..."):
                    st.session_state.target_col = target_col
                    st.session_state.task_type = task_type
                    
                    # Target Column Safety: Drop rows with NaN in target ONLY (Required by PyCaret)
                    initial_count = len(df)
                    df = df.dropna(subset=[target_col])
                    if len(df) < initial_count:
                        st.info(f"ℹ️ Dropped {initial_count - len(df)} row(s) because the Target column '{target_col}' had missing values.")

                    # Check for Task-Type Mismatch (Classification vs Regression)
                    unique_count = df[target_col].nunique()
                    is_numeric = pd.api.types.is_numeric_dtype(df[target_col])
                    
                    if task_type == "Classification" and is_numeric and unique_count > 20:
                        st.warning(f"⚠️ **Caution**: '{target_col}' has {unique_count} unique numeric values. This looks like a **Regression** task. If you see 'unseen labels' errors, please switch to Regression.")
                    elif task_type == "Regression" and not is_numeric:
                        st.warning(f"⚠️ **Caution**: '{target_col}' is non-numeric. Regression requires a numeric target. Please switch to **Classification**.")

                    # Determine strategy based on task and data distribution
                    if task_type == "Classification":
                        fold_strategy = 'stratifiedkfold'
                        data_split_stratify = True
                        class_counts = df[target_col].value_counts()
                        if (class_counts < 2).any():
                            fold_strategy = 'kfold'
                            data_split_stratify = False
                            st.warning("ℹ️ One or more classes have only 1 member. Switching to non-stratified splitting to avoid errors.")
                    else:
                        # Regression defaults
                        fold_strategy = 'kfold'
                        data_split_stratify = False
                    
                    # Prepare setup parameters
                    setup_params = {
                        "data": df,
                        "target": target_col,
                        "session_id": 42,
                        "verbose": False,
                        "numeric_imputation": prep_config.get('numeric_imputation', 'mean'),
                        "normalize": prep_config.get('normalize', False),
                        "normalize_method": prep_config.get('normalize_method', 'zscore'),
                        "remove_outliers": prep_config.get('remove_outliers', False),
                        "feature_selection": prep_config.get('feature_selection', False),
                        "max_encoding_ohe": 20,
                        "fold_strategy": fold_strategy,
                        "data_split_stratify": data_split_stratify
                    }
                    
                    # Only Classification supports fix_imbalance
                    if task_type == "Classification":
                        setup_params["fix_imbalance"] = prep_config.get('fix_imbalance', False)

                    setup(**setup_params)
                    
                with st.spinner("Comparing models..."):
                    best_model = compare_models()
                    st.session_state.best_model = best_model
                    res_df = pull()
                    st.session_state.results_df = res_df
                    # Save baseline metrics from top model
                    st.session_state.baseline_metrics = res_df.iloc[0].to_dict()
                    
                st.success("Baseline Experiment completed!")
                
            except Exception as e:
                if "unseen labels" in str(e).lower():
                    st.error("❌ **Task Mismatch Error**: It looks like you're trying to perform 'Classification' on a continuous numeric column (like Price or Score). Please switch to **Regression** and try again.")
                else:
                    st.error(f"Error during Auto ML: {e}")
                st.code(traceback.format_exc(), language="python")

        if 'best_model' in st.session_state:
            st.dataframe(st.session_state.results_df)
            st.write("Current Best Model:", st.session_state.best_model)

    with tab2:
        if 'best_model' not in st.session_state:
            st.warning("Please run the experiment in the first tab first.")
        else:
            st.subheader("Push Performance to the Limit")
            
            # Helper for displaying evaluation dashboard
            def show_eval_dashboard():
                if 'opt_metrics' in st.session_state:
                    st.markdown('<div class="result-card">', unsafe_allow_html=True)
                    st.markdown("### 🏆 Optimization Success Report")
                    cols = st.columns(len(st.session_state.opt_metrics))
                    baseline = st.session_state.get('baseline_metrics', {})
                    
                    for i, (metric, opt_val) in enumerate(st.session_state.opt_metrics.items()):
                        # We only want to show primary metrics, not 'Model' name
                        if metric == 'Model': continue
                        
                        base_val = baseline.get(metric, 0)
                        try:
                            # Calculate delta for metrics (all metrics like Acc, AUC, R2 are 'higher is better')
                            delta = float(opt_val) - float(base_val)
                            cols[i % len(cols)].metric(label=metric, value=f"{opt_val:.4f}", delta=f"{delta:.4f}")
                        except (ValueError, TypeError):
                            pass
                    st.markdown('</div>', unsafe_allow_html=True)

            col1, col2, col3 = st.columns(3)
            
            with col1:
                if st.button("Tune Model (Bayesian Search)", key="tune_model_btn"):
                    if task_type == "Classification":
                        from pycaret.classification import tune_model, pull
                    else:
                        from pycaret.regression import tune_model, pull
                    with st.spinner("Tuning hyperparameters..."):
                        try:
                            st.session_state.best_model = tune_model(st.session_state.best_model)
                            st.session_state.opt_metrics = pull().iloc[0].to_dict()
                            st.success("Model tuned successfully!")
                        except Exception as e:
                            st.error(f"⚠️ Tuning failed: {e}")
                            st.info("💡 Some models with fixed parameters cannot be tuned. Try Ensembling or Stacking instead.")
            
            with col2:
                if st.button("Ensemble (Bagging/Boosting)", key="ensemble_model_btn"):
                    if task_type == "Classification":
                        from pycaret.classification import ensemble_model, pull
                    else:
                        from pycaret.regression import ensemble_model, pull
                    with st.spinner("Ensembling..."):
                        try:
                            st.session_state.best_model = ensemble_model(st.session_state.best_model)
                            st.session_state.opt_metrics = pull().iloc[0].to_dict()
                            st.success("Ensemble created!")
                        except TypeError as te:
                            if "sample weights" in str(te).lower():
                                st.warning("⚠️ **Model Limitation**: Your current best model (e.g. KNN) does not support sample weights required for Boosting/Bagging.")
                                st.info("💡 **Solution**: Try 'Stacking' or 'Tuning' this model instead.")
                            else: st.error(f"Ensembling error: {te}")
                        except Exception as e:
                            st.error(f"⚠️ Ensembling failed: {e}")

            with col3:
                if st.button("Stack Top 3 Models", key="stack_models_btn"):
                    if task_type == "Classification":
                        from pycaret.classification import compare_models, stack_models, pull
                    else:
                        from pycaret.regression import compare_models, stack_models, pull
                    with st.spinner("Stacking..."):
                        try:
                            top3 = compare_models(n_select=3, verbose=False)
                            st.session_state.best_model = stack_models(top3)
                            st.session_state.opt_metrics = pull().iloc[0].to_dict()
                            st.success("Stacking complete!")
                        except Exception as e:
                            st.error(f"⚠️ Stacking failed: {e}")
                            st.info("💡 Stacking requires a variety of models. If your top 3 models are identical, stacking may not work.")
            
            # Display Dashboard after any optimization button is clicked
            show_eval_dashboard()

    with tab3:
        if 'best_model' not in st.session_state:
            st.warning("Please run experiment first.")
        else:
            st.subheader("Model Visualizations & SHAP")
            plot_type = st.selectbox("Select Insight Type", ["pipeline", "auc", "confusion_matrix", "feature", "learning", "interpret"], key="plot_type_select")
            if st.button("Generate Visualization", key="generate_plot_btn"):
                if plot_type == "interpret":
                    if task_type == "Classification":
                        from pycaret.classification import interpret_model
                    else:
                        from pycaret.regression import interpret_model
                    
                    with st.spinner("Generating SHAP map..."):
                        try:
                            interpret_model(st.session_state.best_model, save=True)
                            found_f = False
                            for f in ['SHAP_Feature_Importance.png', 'interpret.png']:
                                if os.path.exists(f):
                                    st.image(f)
                                    found_f = True
                                    break
                            if not found_f: st.error("Could not find the generated SHAP plot file.")
                        except Exception as e:
                            if "only supports tree based models" in str(e).lower():
                                st.warning("⚠️ **Interpretation Limitation**: SHAP (Interpret) only works for Tree-based models like **Random Forest**, **Decision Trees**, or **LightGBM**.")
                                st.info("💡 **Solution**: Try a different 'Insight Type' like 'Feature' or 'AUC' for this model.")
                            else: st.error(f"Error during interpretation: {e}")
                else:
                    if task_type == "Classification":
                        from pycaret.classification import plot_model
                    else:
                        from pycaret.regression import plot_model
                    
                    with st.spinner("Generating plot..."):
                        try:
                            p = plot_model(st.session_state.best_model, plot=plot_type, display_format='streamlit', save=True)
                            if os.path.exists(p): st.image(p)
                        except TypeError as te:
                            if "feature_importances_" in str(te).lower() or "coef_" in str(te).lower():
                                st.warning(f"⚠️ **Plot Limitation**: The Feature Importance plot is not available for this model (unsupported by {type(st.session_state.best_model).__name__}).")
                                st.info("💡 **Solution**: Try 'Confusion Matrix' (for Classification) or 'AUC' instead.")
                            else: st.error(f"Plot Error: {te}")
                        except Exception as e:
                            st.error(f"⚠️ Error generating plot: {e}")

    with tab4:
        if 'best_model' not in st.session_state:
            st.warning("Please run experiment first.")
        else:
            st.subheader("Prepare for Production")
            if st.button("🚀 Finalize & Save Production Pipeline", key="finalize_model_btn"):
                if task_type == "Classification":
                    from pycaret.classification import finalize_model, save_model
                else:
                    from pycaret.regression import finalize_model, save_model
                with st.spinner("Finalizing..."):
                    st.session_state.best_model = finalize_model(st.session_state.best_model)
                    save_model(st.session_state.best_model, 'best_model_pipeline')
                st.success("Model saved as 'best_model_pipeline.pkl'!")
            
            st.markdown("---")
            c1, c2 = st.columns(2)
            with c1:
                if st.button("Download Training Script", key="download_script_btn"):
                    st.download_button("Get .py Script", generate_training_script(prep_config, st.session_state.target_col, st.session_state.task_type), "train.py", key="finalize_script_btn")
            with c2:
                if st.button("Download FastAPI App", key="download_api_btn"):
                    st.download_button("Get main.py", generate_fastapi_wrapper(), "main.py", key="finalize_api_btn")

def run_anomaly_detection(df: pd.DataFrame):
    """Runs Unsupervised Anomaly Detection using pycaret."""
    st.markdown("### 🕵️ Automated Anomaly Detection")
    if df is None:
        st.warning("Please upload and prepare data first.")
        return

    from pycaret.anomaly import setup, create_model, assign_model, plot_model, predict_model
    
    model_type = st.selectbox("Select Anomaly Detection Algorithm", ["iforest", "knn", "cluster", "svm"], key="anomaly_model_select")
    fraction = st.slider("Contamination Fraction", 0.01, 0.2, 0.05, key="anomaly_fraction_slider")
    
    if st.button("Run Anomaly Detection", key="run_anomaly_btn"):
        # Check for high-cardinality features to avoid memory-error
        high_cardinality = [col for col in df.columns if df[col].nunique() > 100 and df[col].dtype == 'object']
        if high_cardinality:
            st.warning(f"Columns {high_cardinality} have many unique values. If error occurs, please drop them in 'Data Preparation'.")
            
        with st.spinner("Setting up anomaly detection..."):
            setup(data=df, session_id=42, verbose=False, max_encoding_ohe=20)
            model = create_model(model_type, fraction=fraction)
            results = assign_model(model)
            st.session_state.anomaly_results = results
            st.success("Anomaly detection completed!")
            
    if 'anomaly_results' in st.session_state:
        st.markdown("#### Detection Results (Anomaly=1)")
        st.dataframe(st.session_state.anomaly_results.head(100))
        
        csv = st.session_state.anomaly_results.to_csv(index=False).encode('utf-8')
        st.download_button("Download Results with Anomaly Scores", csv, "anomaly_results.csv")

def generate_training_script(prep_config, target, task):
    """Generates a standalone Python script to reproduce the model."""
    module = "classification" if task == "Classification" else "regression"
    
    script = f"""import pandas as pd
from pycaret.{module} import *

# 1. Load Data
df = pd.read_csv('your_data.csv')

# 2. Setup Experiment (Using your GUI settings)
s = setup(
    data=df, 
    target='{target}', 
    session_id=42,
    numeric_imputation='{prep_config.get('numeric_imputation', 'mean')}',
    normalize={prep_config.get('normalize', False)},
    normalize_method='{prep_config.get('normalize_method', 'zscore')}',
    remove_outliers={prep_config.get('remove_outliers', False)},
    fix_imbalance={prep_config.get('fix_imbalance', False)},
    feature_selection={prep_config.get('feature_selection', False)}
)

# 3. Training & Optimization
print("Comparing models...")
best = compare_models()
tuned = tune_model(best)
final_model = finalize_model(tuned)

# 4. Save Model
save_model(final_model, 'best_model_pipeline')
print("Model saved as best_model_pipeline.pkl")
"""
    return script

def generate_fastapi_wrapper():
    """Generates a FastAPI wrapper for the saved model."""
    content = """from fastapi import FastAPI
import pandas as pd
from pycaret.classification import load_model, predict_model
import uvicorn

app = FastAPI(title="AI Model API")
model = load_model('best_model_pipeline')

@app.post("/predict")
def predict(data: dict):
    # Convert input dict to DataFrame
    df = pd.DataFrame([data])
    predictions = predict_model(model, data=df)
    return predictions.to_dict(orient='records')[0]

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
"""
    return content

def prediction_pipeline_interface(df: pd.DataFrame):
    """Interface for making real-time or batch predictions."""
    st.markdown("### 🔮 Automated Prediction Pipeline")
    
    if 'best_model' not in st.session_state:
        st.warning("Please train and finalize a model first in the 'Auto ML Model' tab.")
        # Check if local file exists to load
        if os.path.exists('best_model_pipeline.pkl'):
            if st.button("📂 Load Existing Model (best_model_pipeline.pkl)", key="load_model_btn"):
                try:
                    # Determine task type from session state or default to classification
                    task_type = st.session_state.get('task_type', 'Classification')
                    if task_type == "Classification":
                        from pycaret.classification import load_model
                    else:
                        from pycaret.regression import load_model
                        
                    st.session_state.best_model = load_model('best_model_pipeline')
                    st.success("Model loaded from disk!")
                    st.rerun()
                except Exception as e:
                    st.error(f"Error loading model: {e}")
        return

    predict_mode = st.radio("Prediction Mode", ["Real-time (Manual Input)", "Batch (CSV Upload)"])
    
    # Load model if not in session state but file exists
    if 'best_model' in st.session_state:
        model = st.session_state.best_model
    else:
        st.error("No model found.")
        return

    if predict_mode == "Real-time (Manual Input)":
        st.subheader("Manual Data Input")
        # Identify feature columns (excluding target)
        # We use the original df to get feature names
        features = [col for col in df.columns if col not in st.session_state.get('target_col', '')]
        
        input_data = {}
        cols = st.columns(3)
        for i, feat in enumerate(features):
            with cols[i % 3]:
                if df[feat].dtype == 'object':
                    input_data[feat] = st.selectbox(f"Select {feat}", options=df[feat].unique(), key=f"input_{feat}")
                else:
                    input_data[feat] = st.number_input(f"Enter {feat}", value=float(df[feat].mean()), key=f"input_{feat}")
        
        if st.button("Predict", key="predict_realtime_btn"):
            input_df = pd.DataFrame([input_data])
            # PyCaret predict_model applies the whole pipeline
            from pycaret.classification import predict_model # Import doesn't matter, it's generic enough
            predictions = predict_model(model, data=input_df)
            st.markdown('<div class="result-card">', unsafe_allow_html=True)
            st.markdown("#### 🎯 Prediction Result")
            st.dataframe(predictions)
            st.markdown('</div>', unsafe_allow_html=True)
            
    else:
        st.subheader("Batch Prediction")
        batch_file = st.file_uploader("Upload CSV for Prediction", type=["csv"])
        if batch_file:
            batch_df = pd.read_csv(batch_file)
            st.write("Uploaded Data Preview:")
            st.dataframe(batch_df.head())
            
            if st.button("Run Batch Prediction", key="run_batch_predict_btn"):
                from pycaret.classification import predict_model
                predictions = predict_model(model, data=batch_df)
                st.markdown('<div class="result-card">', unsafe_allow_html=True)
                st.success("✅ Batch Prediction completed!")
                st.dataframe(predictions)
                
                csv = predictions.to_csv(index=False).encode('utf-8')
                st.download_button("Download Predictions", csv, "predictions.csv", "text/csv", key="download_batch_preds_btn")
                st.markdown('</div>', unsafe_allow_html=True)

def chat_with_data_interface(df: pd.DataFrame):
    """Chat interface using PandasAI."""
    
    import os
    env_api_key = os.getenv("GEMINI_API_KEY", "")
    llm_api_key = st.text_input("Enter your Gemini API Key", value=env_api_key, type="password")
    
    if not llm_api_key:
        st.warning("Please enter a Gemini API key to enable chat capabilities.")
        return
        
    try:
        from pandasai import SmartDataframe
        from pandasai.llm import GoogleGemini
        
        @st.cache_resource
        def get_smart_dataframe(df: pd.DataFrame, api_key: str):
            llm = GoogleGemini(api_key=api_key)
            return SmartDataframe(df, config={"llm": llm, "enable_cache": True})
        
        # Monkey-patch info: The patch is now applied at the module level to avoid recursion.
        sdf = get_smart_dataframe(df, llm_api_key)

        
        # Initialize chat history
        if "messages" not in st.session_state:
            st.session_state.messages = []
            
        # Display chat messages from history on app rerun
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
                
        # React to user input
        if prompt := st.chat_input("Ask a question about your data"):
            # Display user message
            st.chat_message("user").markdown(prompt)
            # Add user message to chat history
            st.session_state.messages.append({"role": "user", "content": prompt})

            # Get bot response
            with st.spinner("Thinking..."):
                response = sdf.chat(prompt)
                
            # Display assistant response
            with st.chat_message("assistant"):
                st.markdown(response)
                
            # Add assistant response to chat history
            st.session_state.messages.append({"role": "assistant", "content": str(response)})
            
    except Exception as e:
        st.error(f"Error setting up Chat interface: {e}")
