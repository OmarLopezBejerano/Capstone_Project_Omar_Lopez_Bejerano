import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sys
from pathlib import Path
import numpy as np
from scipy import stats
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import cross_val_score

# parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from utils.database import get_predictions_with_cuff, get_all_predictions, update_calibrated_bp
from utils.calibration import CalibrationManager

st.set_page_config(page_title="Calibration", layout="wide")

# Initialize calibration manager
if 'calibration_manager' not in st.session_state:
    st.session_state.calibration_manager = CalibrationManager()

cal_manager = st.session_state.calibration_manager

st.title("BP Calibration")
st.markdown("Personalize BP predictions using reference cuff measurements.")

# Get predictions with cuff measurements
predictions_with_cuff = get_predictions_with_cuff()

if len(predictions_with_cuff) == 0:
    st.warning("No recordings with cuff measurements yet. Upload recordings and add cuff BP values to enable calibration.")
    st.stop()

df_cuff = pd.DataFrame(predictions_with_cuff)

# Status Panel
st.header("Calibration Status")

status = cal_manager.get_status()

col1, col3, col4 = st.columns(3)

with col1:
    st.metric("Status", "Active" if status['is_fitted'] else "Not Fitted")

with col3:
    st.metric("Training Samples", status['n_samples'])

with col4:
    if status['is_fitted']:
        improvement_sbp = status.get('improvement_sbp') or 0
        improvement_dbp = status.get('improvement_dbp') or 0
        improvement = (improvement_sbp + improvement_dbp) / 2
        st.metric("Avg Improvement", f"{improvement:.1f} mmHg")
    else:
        st.metric("Avg Improvement", "—")

st.divider()

if len(df_cuff) < 3:
    st.warning("Need at least 3 recordings to fit calibration.")
else:
    # Model configuration
    st.subheader("Model Configuration")

    col1, col2, col3 = st.columns(3)

    with col1:
        sbp_degree = st.selectbox(
            "SBP Calibration",
            options=[1, 2, 3],
            format_func=lambda x: "Linear" if x == 1 else f"Polynomial (Degree {x})",
            help="Based on linearity assessment, choose polynomial if R² < 0.5 and you have 5+ points"
        )

    with col2:
        dbp_degree = st.selectbox(
            "DBP Calibration",
            options=[1, 2, 3],
            format_func=lambda x: "Linear" if x == 1 else f"Polynomial (Degree {x})",
            help="Based on linearity assessment, choose polynomial if R² < 0.5 and you have 5+ points"
        )

    with col3:
        st.write("")
        st.write("")
        if st.button("Fit Calibration", type="primary"):
            with st.spinner("Fitting calibration model..."):
                try:
                    result = cal_manager.fit(df_cuff, sbp_degree=sbp_degree, dbp_degree=dbp_degree)

                    if result['success']:
                        st.success(f"Calibration fitted with {result['n_samples']} samples!")

                        # Update all predictions with calibrated values
                        all_predictions = get_all_predictions()
                        df_all = pd.DataFrame(all_predictions)
                        df_calibrated = cal_manager.calibrate_dataframe(df_all)

                        # Save calibrated values to database
                        for idx, row in df_calibrated.iterrows():
                            update_calibrated_bp(
                                int(row['id']),
                                float(row['calibrated_sbp']),
                                float(row['calibrated_dbp']),
                                str(row['calibration_model'])
                            )

                        st.rerun()
                    else:
                        st.error(f"✗ {result['error']}")
                except Exception as e:
                    st.error(f"Error fitting calibration: {str(e)}")
                    st.info("Try clearing the existing calibration first.")

if status['is_fitted']:
    st.success(f"{status['message']}")

    # Show detailed metrics
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Systolic BP")
        st.write(f"**MAE Before:** {status.get('sbp_mae_before', 0):.2f} mmHg")
        st.write(f"**MAE After:** {status.get('sbp_mae_after', 0):.2f} mmHg")
        st.write(f"**Improvement:** {status.get('improvement_sbp', 0):.2f} mmHg")

    with col2:
        st.subheader("Diastolic BP")
        st.write(f"**MAE Before:** {status.get('dbp_mae_before', 0):.2f} mmHg")
        st.write(f"**MAE After:** {status.get('dbp_mae_after', 0):.2f} mmHg")
        st.write(f"**Improvement:** {status.get('improvement_dbp', 0):.2f} mmHg")
else:
    st.info(f"You have {len(df_cuff)} recordings with cuff measurements. Click 'Fit Calibration' below to start.")

st.divider()

# Before/After Comparison
if status['is_fitted']:
    st.header("Before/After Comparison")

    # Calculate errors
    df_cuff['sbp_error_before'] = abs(df_cuff['sbp_mean'] - df_cuff['cuff_sbp'])
    df_cuff['dbp_error_before'] = abs(df_cuff['dbp_mean'] - df_cuff['cuff_dbp'])

    # Get calibrated predictions
    df_calibrated = cal_manager.calibrate_dataframe(df_cuff)
    df_calibrated['sbp_error_after'] = abs(df_calibrated['calibrated_sbp'] - df_calibrated['cuff_sbp'])
    df_calibrated['dbp_error_after'] = abs(df_calibrated['calibrated_dbp'] - df_calibrated['cuff_dbp'])

    # comparison plot
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=("Systolic BP", "Diastolic BP"),
        vertical_spacing=0.15
    )

    # SBP plot
    fig.add_trace(
        go.Box(y=df_cuff['sbp_error_before'], name='Before', marker_color='lightcoral'),
        row=1, col=1
    )
    fig.add_trace(
        go.Box(y=df_calibrated['sbp_error_after'], name='After', marker_color='lightgreen'),
        row=1, col=1
    )

    # DBP plot
    fig.add_trace(
        go.Box(y=df_cuff['dbp_error_before'], name='Before', marker_color='lightcoral', showlegend=False),
        row=1, col=2
    )
    fig.add_trace(
        go.Box(y=df_calibrated['dbp_error_after'], name='After', marker_color='lightgreen', showlegend=False),
        row=1, col=2
    )

    fig.update_yaxes(title_text="Absolute Error (mmHg)", row=1, col=1)
    fig.update_yaxes(title_text="Absolute Error (mmHg)", row=1, col=2)

    fig.update_layout(height=400, showlegend=True)

    st.plotly_chart(fig, use_container_width=True)

    # Scatter comparison
    st.subheader("Prediction Accuracy")

    fig2 = make_subplots(
        rows=1, cols=2,
        subplot_titles=("Systolic BP", "Diastolic BP")
    )

    # SBP scatter
    fig2.add_trace(
        go.Scatter(
            x=df_cuff['cuff_sbp'],
            y=df_cuff['sbp_mean'],
            mode='markers',
            name='Raw',
            marker=dict(color='lightcoral', size=8)
        ),
        row=1, col=1
    )
    fig2.add_trace(
        go.Scatter(
            x=df_calibrated['cuff_sbp'],
            y=df_calibrated['calibrated_sbp'],
            mode='markers',
            name='Calibrated',
            marker=dict(color='lightgreen', size=8)
        ),
        row=1, col=1
    )

    # DBP scatter
    fig2.add_trace(
        go.Scatter(
            x=df_cuff['cuff_dbp'],
            y=df_cuff['dbp_mean'],
            mode='markers',
            name='Raw',
            marker=dict(color='lightcoral', size=8),
            showlegend=False
        ),
        row=1, col=2
    )
    fig2.add_trace(
        go.Scatter(
            x=df_calibrated['cuff_dbp'],
            y=df_calibrated['calibrated_dbp'],
            mode='markers',
            name='Calibrated',
            marker=dict(color='lightgreen', size=8),
            showlegend=False
        ),
        row=1, col=2
    )

    # identity lines
    min_sbp, max_sbp = df_cuff['cuff_sbp'].min() - 10, df_cuff['cuff_sbp'].max() + 10
    min_dbp, max_dbp = df_cuff['cuff_dbp'].min() - 10, df_cuff['cuff_dbp'].max() + 10

    fig2.add_trace(
        go.Scatter(
            x=[min_sbp, max_sbp],
            y=[min_sbp, max_sbp],
            mode='lines',
            line=dict(color='gray', dash='dash'),
            name='Perfect',
            showlegend=True
        ),
        row=1, col=1
    )

    fig2.add_trace(
        go.Scatter(
            x=[min_dbp, max_dbp],
            y=[min_dbp, max_dbp],
            mode='lines',
            line=dict(color='gray', dash='dash'),
            showlegend=False
        ),
        row=1, col=2
    )

    fig2.update_xaxes(title_text="Cuff BP (mmHg)", row=1, col=1)
    fig2.update_xaxes(title_text="Cuff BP (mmHg)", row=1, col=2)
    fig2.update_yaxes(title_text="Predicted BP (mmHg)", row=1, col=1)
    fig2.update_yaxes(title_text="Predicted BP (mmHg)", row=1, col=2)

    fig2.update_layout(height=400)

    st.plotly_chart(fig2, use_container_width=True)

    # Validation Section
    st.divider()
    st.header("Model Validation")
    st.markdown("Statistical analysis to validate calibration effectiveness")

    # Validation metrics
    col1, col2, col3, col4 = st.columns(4)

    # Calculate validation metrics
    raw_sbp_mae = np.mean(df_cuff['sbp_error_before'])
    cal_sbp_mae = np.mean(df_calibrated['sbp_error_after'])
    raw_dbp_mae = np.mean(df_cuff['dbp_error_before'])
    cal_dbp_mae = np.mean(df_calibrated['dbp_error_after'])

    raw_sbp_std = np.std(df_cuff['sbp_error_before'])
    cal_sbp_std = np.std(df_calibrated['sbp_error_after'])
    raw_dbp_std = np.std(df_cuff['dbp_error_before'])
    cal_dbp_std = np.std(df_calibrated['dbp_error_after'])

    with col1:
        st.metric(
            "SBP MAE Reduction",
            f"{((raw_sbp_mae - cal_sbp_mae) / raw_sbp_mae * 100):.1f}%",
            f"{raw_sbp_mae - cal_sbp_mae:.2f} mmHg"
        )

    with col2:
        st.metric(
            "DBP MAE Reduction",
            f"{((raw_dbp_mae - cal_dbp_mae) / raw_dbp_mae * 100):.1f}%",
            f"{raw_dbp_mae - cal_dbp_mae:.2f} mmHg"
        )

    with col3:
        st.metric(
            "SBP Std Dev",
            f"{cal_sbp_std:.2f} mmHg",
            f"{raw_sbp_std - cal_sbp_std:.2f} mmHg",
            delta_color="inverse"
        )

    with col4:
        st.metric(
            "DBP Std Dev",
            f"{cal_dbp_std:.2f} mmHg",
            f"{raw_dbp_std - cal_dbp_std:.2f} mmHg",
            delta_color="inverse"
        )

    # Statistical significance testing
    st.subheader("Statistical Significance")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Systolic BP**")

        # Paired t-test
        from scipy.stats import ttest_rel
        t_stat_sbp, p_value_sbp = ttest_rel(df_cuff['sbp_error_before'], df_calibrated['sbp_error_after'])

        st.write(f"**Paired t-test**: t = {t_stat_sbp:.3f}, p = {p_value_sbp:.4f}")

        if p_value_sbp < 0.001:
            st.success("Highly significant improvement (p < 0.001)")
        elif p_value_sbp < 0.05:
            st.success("Significant improvement (p < 0.05)")
        elif p_value_sbp < 0.1:
            st.info("Marginally significant (p < 0.1)")
        else:
            st.warning("Not statistically significant (p ≥ 0.1)")

        # Effect size (Cohen's d)
        mean_diff = np.mean(df_cuff['sbp_error_before'] - df_calibrated['sbp_error_after'])
        pooled_std = np.sqrt((raw_sbp_std**2 + cal_sbp_std**2) / 2)
        cohens_d_sbp = mean_diff / pooled_std

        st.write(f"**Effect Size (Cohen's d)**: {cohens_d_sbp:.3f}")

        if abs(cohens_d_sbp) > 0.8:
            st.success("Large effect size")
        elif abs(cohens_d_sbp) > 0.5:
            st.info("Medium effect size")
        elif abs(cohens_d_sbp) > 0.2:
            st.info("Small effect size")
        else:
            st.warning("Negligible effect size")

    with col2:
        st.markdown("**Diastolic BP**")

        # Paired t-test
        t_stat_dbp, p_value_dbp = ttest_rel(df_cuff['dbp_error_before'], df_calibrated['dbp_error_after'])

        st.write(f"**Paired t-test**: t = {t_stat_dbp:.3f}, p = {p_value_dbp:.4f}")

        if p_value_dbp < 0.001:
            st.success("Highly significant improvement (p < 0.001)")
        elif p_value_dbp < 0.05:
            st.success("Significant improvement (p < 0.05)")
        elif p_value_dbp < 0.1:
            st.info("Marginally significant (p < 0.1)")
        else:
            st.warning("Not statistically significant (p ≥ 0.1)")

        # Effect size (Cohen's d)
        mean_diff = np.mean(df_cuff['dbp_error_before'] - df_calibrated['dbp_error_after'])
        pooled_std = np.sqrt((raw_dbp_std**2 + cal_dbp_std**2) / 2)
        cohens_d_dbp = mean_diff / pooled_std

        st.write(f"**Effect Size (Cohen's d)**: {cohens_d_dbp:.3f}")

        if abs(cohens_d_dbp) > 0.8:
            st.success("Large effect size")
        elif abs(cohens_d_dbp) > 0.5:
            st.info("Medium effect size")
        elif abs(cohens_d_dbp) > 0.2:
            st.info("Small effect size")
        else:
            st.warning("Negligible effect size")

    # Clinical validation standards
    st.subheader("Clinical Validation Standards")
    st.markdown("""
    **IEEE Standard for Wearable Cuffless BP Devices (IEEE 1708a-2019)**:
    - Grade A: MAE ≤ 5 mmHg, Std Dev ≤ 8 mmHg
    - Grade B: MAE ≤ 10 mmHg, Std Dev ≤ 12 mmHg
    - Grade C: MAE ≤ 15 mmHg, Std Dev ≤ 15 mmHg
    """)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Systolic BP After Calibration**")

        if cal_sbp_mae <= 5 and cal_sbp_std <= 8:
            grade = "A"
            color = "success"
            st.success(f"**Grade A** (MAE: {cal_sbp_mae:.2f}, SD: {cal_sbp_std:.2f})")
        elif cal_sbp_mae <= 10 and cal_sbp_std <= 12:
            grade = "B"
            color = "info"
            st.info(f"**Grade B** (MAE: {cal_sbp_mae:.2f}, SD: {cal_sbp_std:.2f})")
        elif cal_sbp_mae <= 15 and cal_sbp_std <= 15:
            grade = "C"
            color = "warning"
            st.warning(f"**Grade C** (MAE: {cal_sbp_mae:.2f}, SD: {cal_sbp_std:.2f})")
        else:
            grade = "Below C"
            color = "error"
            st.error(f"✗ **Below Grade C** (MAE: {cal_sbp_mae:.2f}, SD: {cal_sbp_std:.2f})")

    with col2:
        st.markdown("**Diastolic BP After Calibration**")

        if cal_dbp_mae <= 5 and cal_dbp_std <= 8:
            grade = "A"
            color = "success"
            st.success(f"**Grade A** (MAE: {cal_dbp_mae:.2f}, SD: {cal_dbp_std:.2f})")
        elif cal_dbp_mae <= 10 and cal_dbp_std <= 12:
            grade = "B"
            color = "info"
            st.info(f"**Grade B** (MAE: {cal_dbp_mae:.2f}, SD: {cal_dbp_std:.2f})")
        elif cal_dbp_mae <= 15 and cal_dbp_std <= 15:
            grade = "C"
            color = "warning"
            st.warning(f"**Grade C** (MAE: {cal_dbp_mae:.2f}, SD: {cal_dbp_std:.2f})")
        else:
            grade = "Below C"
            color = "error"
            st.error(f"✗ **Below Grade C** (MAE: {cal_dbp_mae:.2f}, SD: {cal_dbp_std:.2f})")

    # Error distribution analysis
    st.subheader("Error Distribution Analysis")

    fig_dist = make_subplots(
        rows=1, cols=2,
        subplot_titles=("Systolic BP Error Distribution", "Diastolic BP Error Distribution")
    )

    # SBP distribution
    fig_dist.add_trace(
        go.Histogram(
            x=df_cuff['sbp_error_before'],
            name='Before',
            marker_color='lightcoral',
            opacity=0.7,
            nbinsx=10
        ),
        row=1, col=1
    )
    fig_dist.add_trace(
        go.Histogram(
            x=df_calibrated['sbp_error_after'],
            name='After',
            marker_color='lightgreen',
            opacity=0.7,
            nbinsx=10
        ),
        row=1, col=1
    )

    # DBP distribution
    fig_dist.add_trace(
        go.Histogram(
            x=df_cuff['dbp_error_before'],
            name='Before',
            marker_color='lightcoral',
            opacity=0.7,
            nbinsx=10,
            showlegend=False
        ),
        row=1, col=2
    )
    fig_dist.add_trace(
        go.Histogram(
            x=df_calibrated['dbp_error_after'],
            name='After',
            marker_color='lightgreen',
            opacity=0.7,
            nbinsx=10,
            showlegend=False
        ),
        row=1, col=2
    )

    fig_dist.update_xaxes(title_text="Absolute Error (mmHg)", row=1, col=1)
    fig_dist.update_xaxes(title_text="Absolute Error (mmHg)", row=1, col=2)
    fig_dist.update_yaxes(title_text="Count", row=1, col=1)
    fig_dist.update_yaxes(title_text="Count", row=1, col=2)

    fig_dist.update_layout(height=400, barmode='overlay')

    st.plotly_chart(fig_dist, use_container_width=True)

    # Bland-Altman Plot
    st.subheader("Bland-Altman Analysis")
    st.markdown("Visualizes agreement between calibrated predictions and cuff measurements")

    fig_ba = make_subplots(
        rows=1, cols=2,
        subplot_titles=("Systolic BP", "Diastolic BP")
    )

    # SBP Bland-Altman
    sbp_mean = (df_calibrated['calibrated_sbp'] + df_calibrated['cuff_sbp']) / 2
    sbp_diff = df_calibrated['calibrated_sbp'] - df_calibrated['cuff_sbp']
    sbp_mean_diff = sbp_diff.mean()
    sbp_std_diff = sbp_diff.std()

    fig_ba.add_trace(
        go.Scatter(
            x=sbp_mean,
            y=sbp_diff,
            mode='markers',
            marker=dict(color='royalblue', size=8),
            name='SBP',
            showlegend=False
        ),
        row=1, col=1
    )

    # Add mean and LoA lines for SBP
    fig_ba.add_hline(y=sbp_mean_diff, line_dash="solid", line_color="green",
                     annotation_text=f"Mean: {sbp_mean_diff:.2f}", row=1, col=1)
    fig_ba.add_hline(y=sbp_mean_diff + 1.96*sbp_std_diff, line_dash="dash", line_color="red",
                     annotation_text=f"+1.96 SD: {sbp_mean_diff + 1.96*sbp_std_diff:.2f}", row=1, col=1)
    fig_ba.add_hline(y=sbp_mean_diff - 1.96*sbp_std_diff, line_dash="dash", line_color="red",
                     annotation_text=f"-1.96 SD: {sbp_mean_diff - 1.96*sbp_std_diff:.2f}", row=1, col=1)

    # DBP Bland-Altman
    dbp_mean = (df_calibrated['calibrated_dbp'] + df_calibrated['cuff_dbp']) / 2
    dbp_diff = df_calibrated['calibrated_dbp'] - df_calibrated['cuff_dbp']
    dbp_mean_diff = dbp_diff.mean()
    dbp_std_diff = dbp_diff.std()

    fig_ba.add_trace(
        go.Scatter(
            x=dbp_mean,
            y=dbp_diff,
            mode='markers',
            marker=dict(color='royalblue', size=8),
            name='DBP',
            showlegend=False
        ),
        row=1, col=2
    )

    # Add mean and LoA lines for DBP
    fig_ba.add_hline(y=dbp_mean_diff, line_dash="solid", line_color="green",
                     annotation_text=f"Mean: {dbp_mean_diff:.2f}", row=1, col=2)
    fig_ba.add_hline(y=dbp_mean_diff + 1.96*dbp_std_diff, line_dash="dash", line_color="red",
                     annotation_text=f"+1.96 SD: {dbp_mean_diff + 1.96*dbp_std_diff:.2f}", row=1, col=2)
    fig_ba.add_hline(y=dbp_mean_diff - 1.96*dbp_std_diff, line_dash="dash", line_color="red",
                     annotation_text=f"-1.96 SD: {dbp_mean_diff - 1.96*dbp_std_diff:.2f}", row=1, col=2)

    fig_ba.update_xaxes(title_text="Mean BP (mmHg)", row=1, col=1)
    fig_ba.update_xaxes(title_text="Mean BP (mmHg)", row=1, col=2)
    fig_ba.update_yaxes(title_text="Difference (mmHg)", row=1, col=1)
    fig_ba.update_yaxes(title_text="Difference (mmHg)", row=1, col=2)

    fig_ba.update_layout(height=400)

    st.plotly_chart(fig_ba, use_container_width=True)

else:
    st.info("Fit calibration to see validation metrics.")
