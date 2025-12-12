import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
from pathlib import Path
import tempfile

sys.path.append(str(Path(__file__).parent.parent.parent))

from utils.database import (
    get_all_predictions,
    save_prediction,
    delete_prediction,
    get_predictions_with_cuff,
    update_calibrated_bp
)
from utils.processing import process_fit_file, get_bp_category
from utils.calibration import CalibrationManager

st.set_page_config(page_title="Dashboard",layout="wide")

# Initialize calibration manager
if 'calibration_manager' not in st.session_state:
    st.session_state.calibration_manager = CalibrationManager()

cal_manager = st.session_state.calibration_manager
cal_status = cal_manager.get_status()

st.title("Dashboard")

# Calibration checkbox
if cal_status['is_fitted']:
    show_calibrated = st.checkbox("Show Calibrated Values", value=True, help="Display calibrated predictions instead of raw predictions")
else:
    show_calibrated = False

# Upload section
with st.expander("Upload New Recording"):
    col1, col2 = st.columns([3, 1])
    with col1:
        uploaded_file = st.file_uploader("Choose .fit file", type=['fit'], label_visibility="collapsed")

    if uploaded_file is not None:
        if 'default_datetime' not in st.session_state:
            st.session_state.default_datetime = datetime.now().strftime("%Y-%m-%d %H:%M")
        if 'cuff_readings' not in st.session_state:
            st.session_state.cuff_readings = []

        col1, col2 = st.columns([3, 2])

        with col1:
            recording_datetime = st.text_input(
                "Recording Date & Time",
                value=st.session_state.default_datetime,
                help="Format: YYYY-MM-DD HH:MM",
                key="recording_datetime_input"
            )
        with col2:
            st.write("")
            st.write("")
            analyze_btn = st.button("Analyze", type="primary", use_container_width=True)

        # Cuff BP Readings Section
        st.subheader("Cuff BP Readings")
        st.caption("Add one or more cuff readings taken during the recording")

        col1, col2, col3 = st.columns([2, 2, 1])
        with col1:
            new_cuff_sbp = st.number_input("SBP", min_value=50, max_value=250, value=120, step=1, key="new_sbp")
        with col2:
            new_cuff_dbp = st.number_input("DBP", min_value=30, max_value=150, value=80, step=1, key="new_dbp")
        with col3:
            st.write("")
            st.write("")
            if st.button("Add Reading"):
                st.session_state.cuff_readings.append({
                    'sbp': new_cuff_sbp,
                    'dbp': new_cuff_dbp
                })
                st.rerun()

        # current readings
        if st.session_state.cuff_readings:
            st.write("**Readings to use:**")
            for idx, reading in enumerate(st.session_state.cuff_readings):
                col1, col2 = st.columns([4, 1])
                with col1:
                    st.write(f"{idx+1}. SBP: {reading['sbp']} mmHg, DBP: {reading['dbp']} mmHg")
                with col2:
                    if st.button("Delete", key=f"del_{idx}"):
                        st.session_state.cuff_readings.pop(idx)
                        st.rerun()

            # average
            avg_sbp = sum(r['sbp'] for r in st.session_state.cuff_readings) / len(st.session_state.cuff_readings)
            avg_dbp = sum(r['dbp'] for r in st.session_state.cuff_readings) / len(st.session_state.cuff_readings)
            st.info(f"Average: {avg_sbp:.0f}/{avg_dbp:.0f} mmHg (will be used for validation)")
        else:
            st.caption("No cuff readings added (optional)")

        # aditional metadata
        st.divider()
        col1, col2, col3 = st.columns(3)
        with col1:
            position = st.selectbox("Position", ["Seated", "Lying Down", "Standing"], index=0)
        with col2:
            post_exercise = st.selectbox("Post-Exercise", ["No", "Light (<30 min ago)", "Moderate (<30 min ago)", "Intense (<30 min ago)"], index=0)
        with col3:
            notes = st.text_input("Notes (optional)", placeholder="Any observations...")

        if analyze_btn:
            # Parse datetime
            try:
                recording_timestamp = datetime.strptime(recording_datetime, "%Y-%m-%d %H:%M")
            except ValueError:
                st.error("Invalid date/time format. Use: YYYY-MM-DD HH:MM")
                st.stop()

            with st.spinner("Processing..."):
                with tempfile.NamedTemporaryFile(delete=False, suffix='.fit') as tmp_file:
                    tmp_file.write(uploaded_file.getvalue())
                    tmp_file_path = tmp_file.name

                result = process_fit_file(tmp_file_path)

                if result['success']:
                    category = get_bp_category(result['sbp_mean'], result['dbp_mean'])

                    #  average cuff readings, if some exist
                    cuff_sbp = None
                    cuff_dbp = None
                    if st.session_state.cuff_readings:
                        cuff_sbp = sum(r['sbp'] for r in st.session_state.cuff_readings) / len(st.session_state.cuff_readings)
                        cuff_dbp = sum(r['dbp'] for r in st.session_state.cuff_readings) / len(st.session_state.cuff_readings)

                    # Apply calibration, if available
                    calibrated_sbp = None
                    calibrated_dbp = None
                    calibration_model = None
                    if cal_status['is_fitted']:
                        record_context = {
                            'sbp_mean': result['sbp_mean'],
                            'dbp_mean': result['dbp_mean'],
                            'mean_hr': result['mean_hr'],
                            'hrv_sdnn': result['hrv_sdnn'],
                            'time_of_day': None,  # calculated in database
                            'position': position,
                            'post_exercise': post_exercise
                        }
                        calibrated_sbp, calibrated_dbp = cal_manager.predict(
                            result['sbp_mean'],
                            result['dbp_mean'],
                            None
                        )
                        calibration_model = cal_status['active_model']

                    save_prediction(
                        filename=uploaded_file.name,
                        timestamp=recording_timestamp,
                        sbp_mean=result['sbp_mean'],
                        sbp_std=result['sbp_std'],
                        sbp_min=result['sbp_min'],
                        sbp_max=result['sbp_max'],
                        dbp_mean=result['dbp_mean'],
                        dbp_std=result['dbp_std'],
                        dbp_min=result['dbp_min'],
                        dbp_max=result['dbp_max'],
                        num_windows=result['num_windows'],
                        duration=result['total_duration'],
                        mean_hr=result['mean_hr'],
                        hrv_sdnn=result['hrv_sdnn'],
                        category=category,
                        cuff_sbp=cuff_sbp,
                        cuff_dbp=cuff_dbp,
                        window_sbp=result.get('window_predictions', {}).get('sbp'),
                        window_dbp=result.get('window_predictions', {}).get('dbp'),
                        position=position,
                        post_exercise=post_exercise,
                        notes=notes if notes else None,
                        calibrated_sbp=calibrated_sbp,
                        calibrated_dbp=calibrated_dbp,
                        calibration_model=calibration_model
                    )

                    st.success("Saved!")

                    # Prompt for recalibration if cuff reading is added
                    if cuff_sbp is not None:
                        predictions_with_cuff = get_predictions_with_cuff()
                        if len(predictions_with_cuff) >= 3:
                            st.info(f"You now have {len(predictions_with_cuff)} recordings with cuff measurements. Consider recalibrating on the Calibration page.")

                    st.session_state.default_datetime = datetime.now().strftime("%Y-%m-%d %H:%M")
                    st.session_state.cuff_readings = []
                    st.rerun()
                else:
                    st.error(f"{result.get('error', 'Error')}")

                try:
                    Path(tmp_file_path).unlink()
                except:
                    pass
try:
    predictions = get_all_predictions()

    if not predictions:
        st.info("No data. Upload your first recording.")
        st.stop()

    df = pd.DataFrame(predictions)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values('timestamp', ascending=False)

    # Apply calibration, if showing calibrated values
    if show_calibrated and cal_status['is_fitted']:
        df = cal_manager.calibrate_dataframe(df)

    # stats
    col1, col2, col3, col4, col5, col6 = st.columns(6)
    with col1:
        st.metric("Recordings", len(df))
    with col2:
        if show_calibrated and 'calibrated_sbp' in df.columns:
            avg_val = df['calibrated_sbp'].dropna().mean()
            if pd.notna(avg_val):
                st.metric("Avg SBP", f"{avg_val:.0f}", help="Calibrated")
            else:
                st.metric("Avg SBP", f"{df['sbp_mean'].mean():.0f}")
        else:
            st.metric("Avg SBP", f"{df['sbp_mean'].mean():.0f}")
    with col3:
        if show_calibrated and 'calibrated_dbp' in df.columns:
            avg_val = df['calibrated_dbp'].dropna().mean()
            if pd.notna(avg_val):
                st.metric("Avg DBP", f"{avg_val:.0f}", help="Calibrated")
            else:
                st.metric("Avg DBP", f"{df['dbp_mean'].mean():.0f}")
        else:
            st.metric("Avg DBP", f"{df['dbp_mean'].mean():.0f}")
    with col4:
        st.metric("Avg HR", f"{df['mean_hr'].mean():.0f}")
    with col5:
        has_cuff = df[(df['cuff_sbp'].notna()) & (df['cuff_dbp'].notna())].shape[0]
        st.metric("w/ Cuff", has_cuff)
    with col6:
        if has_cuff > 0:
            df_with_cuff = df[(df['cuff_sbp'].notna()) & (df['cuff_dbp'].notna())].copy()
            if show_calibrated and 'calibrated_sbp' in df_with_cuff.columns:
                # Drop NaN values before calculating error
                valid_cuff = df_with_cuff[df_with_cuff['calibrated_sbp'].notna() & df_with_cuff['calibrated_dbp'].notna()]
                if len(valid_cuff) > 0:
                    avg_sbp_error = (valid_cuff['calibrated_sbp'] - valid_cuff['cuff_sbp']).abs().mean()
                else:
                    avg_sbp_error = (df_with_cuff['sbp_mean'] - df_with_cuff['cuff_sbp']).abs().mean()
            else:
                avg_sbp_error = (df_with_cuff['sbp_mean'] - df_with_cuff['cuff_sbp']).abs().mean()
            st.metric("Avg Error", f"{avg_sbp_error:.1f}")
        else:
            st.metric("Avg Error", "—")

    st.divider()

    col1, col2 = st.columns([1, 4])
    with col1:
        time_filter = st.selectbox("Period", ["All", "7d", "30d", "90d"], label_visibility="collapsed")

    filtered_df = df.copy()
    if time_filter == "7d":
        filtered_df = df[df['timestamp'] >= datetime.now() - timedelta(days=7)]
    elif time_filter == "30d":
        filtered_df = df[df['timestamp'] >= datetime.now() - timedelta(days=30)]
    elif time_filter == "90d":
        filtered_df = df[df['timestamp'] >= datetime.now() - timedelta(days=90)]

    if filtered_df.empty:
        st.warning("No data in selected period.")
        st.stop()

    # BP Trends
    import plotly.graph_objects as go

    fig = go.Figure()

    # Determine which values to plot
    if show_calibrated and 'calibrated_sbp' in filtered_df.columns:
        sbp_col = 'calibrated_sbp'
        dbp_col = 'calibrated_dbp'
        label_prefix = 'Cal'
    else:
        sbp_col = 'sbp_mean'
        dbp_col = 'dbp_mean'
        label_prefix = 'Pred'

    # BP traces
    if show_calibrated and 'calibrated_sbp' in filtered_df.columns:
        # Calibrated (no error bars)
        fig.add_trace(go.Scatter(
            x=filtered_df['timestamp'],
            y=filtered_df[sbp_col],
            mode='lines+markers',
            name=f'{label_prefix} SBP',
            line=dict(color='#d62728', width=1.5),
            marker=dict(size=5)
        ))
        fig.add_trace(go.Scatter(
            x=filtered_df['timestamp'],
            y=filtered_df[dbp_col],
            mode='lines+markers',
            name=f'{label_prefix} DBP',
            line=dict(color='#1f77b4', width=1.5),
            marker=dict(size=5)
        ))
    else:
        # Raw with error bars
        fig.add_trace(go.Scatter(
            x=filtered_df['timestamp'],
            y=filtered_df[sbp_col],
            error_y=dict(type='data', array=filtered_df['sbp_std'], visible=True, thickness=1, width=2),
            mode='lines+markers',
            name=f'{label_prefix} SBP',
            line=dict(color='#d62728', width=1.5),
            marker=dict(size=5)
        ))
        fig.add_trace(go.Scatter(
            x=filtered_df['timestamp'],
            y=filtered_df[dbp_col],
            error_y=dict(type='data', array=filtered_df['dbp_std'], visible=True, thickness=1, width=2),
            mode='lines+markers',
            name=f'{label_prefix} DBP',
            line=dict(color='#1f77b4', width=1.5),
            marker=dict(size=5)
        ))

    # Cuff BP
    df_with_cuff = filtered_df[(filtered_df['cuff_sbp'].notna()) & (filtered_df['cuff_dbp'].notna())]
    if not df_with_cuff.empty:
        fig.add_trace(go.Scatter(
            x=df_with_cuff['timestamp'],
            y=df_with_cuff['cuff_sbp'],
            mode='markers',
            name='Cuff SBP',
            marker=dict(size=10, color='#d62728', symbol='x', line=dict(width=2))
        ))
        fig.add_trace(go.Scatter(
            x=df_with_cuff['timestamp'],
            y=df_with_cuff['cuff_dbp'],
            mode='markers',
            name='Cuff DBP',
            marker=dict(size=10, color='#1f77b4', symbol='x', line=dict(width=2))
        ))

    # Reference zones
    fig.add_hrect(y0=0, y1=80, fillcolor="green", opacity=0.05, line_width=0)
    fig.add_hrect(y0=120, y1=130, fillcolor="yellow", opacity=0.05, line_width=0)
    fig.add_hrect(y0=130, y1=140, fillcolor="orange", opacity=0.05, line_width=0)
    fig.add_hrect(y0=140, y1=200, fillcolor="red", opacity=0.05, line_width=0)

    fig.update_layout(
        height=400,
        margin=dict(l=40, r=40, t=40, b=40),
        hovermode='x unified',
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    fig.update_xaxes(title_text="Date")
    fig.update_yaxes(title_text="BP (mmHg)")

    st.plotly_chart(fig, use_container_width=True)

    st.divider()

    # Recording list with details
    st.subheader("Recordings")

    col1, col2 = st.columns([3, 1])
    with col1:
        selected_idx = st.selectbox(
            "Select recording:",
            options=range(len(filtered_df)),
            format_func=lambda i: f"{filtered_df.iloc[i]['timestamp'].strftime('%m/%d %H:%M')} - {filtered_df.iloc[i]['filename'][:30]}",
            key="recording_selector"
        )
    with col2:
        st.write("")
        st.write("")
        if st.button("Delete", use_container_width=True, key="delete_btn"):
            record_id = int(filtered_df.iloc[selected_idx]['id'])
            delete_prediction(record_id)
            st.rerun()

    # Show details
    if selected_idx is not None:
        selected = filtered_df.iloc[selected_idx]

        with st.container(border=True):
            st.markdown(f"### {selected['filename']}")
            st.caption(f"{selected['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}")

            col1, col2, col3, col4, col5, col6 = st.columns(6)

            with col1:
                if show_calibrated and pd.notna(selected.get('calibrated_sbp')):
                    st.metric("Cal SBP", f"{selected['calibrated_sbp']:.0f}")
                    st.caption(f"Raw: {selected['sbp_mean']:.0f}")
                else:
                    st.metric("Pred SBP", f"{selected['sbp_mean']:.0f}")
                    st.caption(f"±{selected['sbp_std']:.0f}")
            with col2:
                if show_calibrated and pd.notna(selected.get('calibrated_dbp')):
                    st.metric("Cal DBP", f"{selected['calibrated_dbp']:.0f}")
                    st.caption(f"Raw: {selected['dbp_mean']:.0f}")
                else:
                    st.metric("Pred DBP", f"{selected['dbp_mean']:.0f}")
                    st.caption(f"±{selected['dbp_std']:.0f}")
            with col3:
                if pd.notna(selected['cuff_sbp']):
                    st.metric("Cuff SBP", f"{selected['cuff_sbp']:.0f}")
                    if show_calibrated and pd.notna(selected.get('calibrated_sbp')):
                        error = abs(selected['calibrated_sbp'] - selected['cuff_sbp'])
                    else:
                        error = abs(selected['sbp_mean'] - selected['cuff_sbp'])
                    st.caption(f"Err: {error:.1f}")
                else:
                    st.metric("Cuff SBP", "—")
            with col4:
                if pd.notna(selected['cuff_dbp']):
                    st.metric("Cuff DBP", f"{selected['cuff_dbp']:.0f}")
                    if show_calibrated and pd.notna(selected.get('calibrated_dbp')):
                        error = abs(selected['calibrated_dbp'] - selected['cuff_dbp'])
                    else:
                        error = abs(selected['dbp_mean'] - selected['cuff_dbp'])
                    st.caption(f"Err: {error:.1f}")
                else:
                    st.metric("Cuff DBP", "—")
            with col5:
                st.metric("HR", f"{selected['mean_hr']:.0f}")
                st.caption(f"HRV: {selected['hrv_sdnn']:.0f}")
            with col6:
                st.metric("Category", "")
                st.caption(selected['category'])

            st.caption(f"Duration: {selected['duration']/60:.1f} min | Windows: {selected['num_windows']}")

            # BP plot for the recording
            if pd.notna(selected.get('window_sbp')) and pd.notna(selected.get('window_dbp')):
                import json

                window_sbp = json.loads(selected['window_sbp'])
                window_dbp = json.loads(selected['window_dbp'])

                # Create time axis
                num_windows = len(window_sbp)
                time_points = np.linspace(0, selected['duration'], num_windows)

                fig_detail = go.Figure()

                fig_detail.add_trace(go.Scatter(
                    x=time_points,
                    y=window_sbp,
                    mode='markers+lines',
                    name='Pred SBP',
                    line=dict(color='#d62728', width=1.5),
                    marker=dict(size=6)
                ))

                fig_detail.add_trace(go.Scatter(
                    x=time_points,
                    y=window_dbp,
                    mode='markers+lines',
                    name='Pred DBP',
                    line=dict(color='#1f77b4', width=1.5),
                    marker=dict(size=6)
                ))

                # Add cuff BP as horizontal lines if present
                if pd.notna(selected['cuff_sbp']):
                    fig_detail.add_hline(
                        y=selected['cuff_sbp'],
                        line_dash="solid",
                        line_color="#d62728",
                        line_width=2.5,
                        annotation_text=f"Cuff SBP: {selected['cuff_sbp']:.0f}",
                        annotation_position="right"
                    )

                if pd.notna(selected['cuff_dbp']):
                    fig_detail.add_hline(
                        y=selected['cuff_dbp'],
                        line_dash="solid",
                        line_color="#1f77b4",
                        line_width=2.5,
                        annotation_text=f"Cuff DBP: {selected['cuff_dbp']:.0f}",
                        annotation_position="right"
                    )

                fig_detail.update_layout(
                    height=350,
                    margin=dict(l=20, r=20, t=20, b=20),
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                    xaxis_title="Time (seconds)",
                    yaxis_title="BP (mmHg)"
                )

                st.plotly_chart(fig_detail, use_container_width=True)
            else:
                st.info("Window-level predictions not available for this recording.")

    # # Export
    # csv = filtered_df.to_csv(index=False)
    # st.download_button(
    #     "Export CSV",
    #     csv,
    #     f"bp_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
    #     "text/csv"
    # )

except Exception as e:
    st.error(f"Error: {e}")
    st.exception(e)
