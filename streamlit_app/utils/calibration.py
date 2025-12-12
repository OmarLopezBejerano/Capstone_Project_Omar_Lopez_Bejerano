"""
Calibration Utility Module
===========================
Streamlit integration for personalized BP calibration.
"""

import sys
from pathlib import Path
import pandas as pd
from typing import Optional, Tuple, Dict

# Add calibration module to path
CALIBRATION_DIR = Path(__file__).parent.parent.parent / "calibration"
sys.path.insert(0, str(CALIBRATION_DIR))

from calibration_models import LinearCalibration, HybridCalibration


class CalibrationManager:
    """Manages calibration state and operations for Streamlit app."""

    def __init__(self, model_dir: str = "streamlit_app/calibration_models"):
        """
        Initialize calibration manager.

        Args:
            model_dir: Directory to store calibration models
        """
        self.model_dir = Path(__file__).parent.parent / "calibration_models"
        self.model_dir.mkdir(exist_ok=True)

        self.hybrid_cal = HybridCalibration()
        self.is_fitted = False
        self._load_existing_model()

    @property
    def linear_cal(self):
        """Provide access to linear calibration for backward compatibility."""
        return self.hybrid_cal.linear_cal

    def _load_existing_model(self):
        """Load existing calibration model if available."""
        try:
            success = self.hybrid_cal.load(str(self.model_dir))
            if success and self.hybrid_cal.linear_cal.is_fitted:
                self.is_fitted = True
        except Exception as e:
            print(f"Error loading calibration: {e}")

    def fit(self, predictions_df: pd.DataFrame, sbp_degree: int = 1, dbp_degree: int = 1) -> Dict:
        """
        Fit calibration model on predictions with cuff measurements.

        Args:
            predictions_df: DataFrame with columns including sbp_mean, dbp_mean,
                          cuff_sbp, cuff_dbp, and metadata
            sbp_degree: Polynomial degree for SBP (1=linear, 2-3=polynomial)
            dbp_degree: Polynomial degree for DBP (1=linear, 2-3=polynomial)

        Returns:
            Dictionary with fit results
        """
        # Filter to only predictions with cuff measurements
        df_with_cuff = predictions_df[
            (predictions_df['cuff_sbp'].notna()) &
            (predictions_df['cuff_dbp'].notna())
        ].copy()

        if len(df_with_cuff) < 3:
            return {
                'success': False,
                'error': f'Need at least 3 recordings with cuff measurements (have {len(df_with_cuff)})'
            }

        # Fit the hybrid calibration model
        result = self.hybrid_cal.fit(df_with_cuff, sbp_degree=sbp_degree, dbp_degree=dbp_degree)

        if result['success']:
            # Save the fitted models
            self.hybrid_cal.save(str(self.model_dir))
            self.is_fitted = True

        return result

    def predict(self, sbp: float, dbp: float, record: Optional[Dict] = None) -> Tuple[float, float]:
        """
        Apply calibration to BP prediction.

        Args:
            sbp: Predicted systolic BP
            dbp: Predicted diastolic BP
            record: Optional (not used, kept for compatibility)

        Returns:
            Tuple of (calibrated_sbp, calibrated_dbp)
        """
        if not self.is_fitted:
            return sbp, dbp

        return self.hybrid_cal.predict(sbp, dbp, record)

    def get_status(self) -> Dict:
        """
        Get current calibration status.

        Returns:
            Dictionary with calibration info
        """
        if not self.is_fitted:
            return {
                'is_fitted': False,
                'active_model': 'none',
                'n_samples': 0,
                'message': 'No calibration fitted yet'
            }

        info = self.hybrid_cal.get_info()

        # Build active model description
        sbp_type = info.get('sbp_model_type', 'linear')
        dbp_type = info.get('dbp_model_type', 'linear')

        if sbp_type == dbp_type:
            active_model_desc = sbp_type
        else:
            active_model_desc = f"hybrid (SBP:{sbp_type}, DBP:{dbp_type})"

        # Get metrics from the appropriate model(s)
        # If using polynomial for either component, prefer poly_metrics
        # Otherwise use linear_metrics
        linear_metrics = info.get('linear_metrics', {})
        poly_metrics = info.get('poly_metrics', {})

        # Use polynomial metrics if available, otherwise fall back to linear
        if poly_metrics:
            metrics = poly_metrics
        else:
            metrics = linear_metrics

        return {
            'is_fitted': True,
            'active_model': active_model_desc,
            'sbp_model_type': sbp_type,
            'dbp_model_type': dbp_type,
            'n_samples': info.get('n_samples', 0),
            'sbp_mae_before': metrics.get('sbp_mae_before', 0),
            'sbp_mae_after': metrics.get('sbp_mae_after', 0),
            'improvement_sbp': metrics.get('improvement_sbp', 0),
            'dbp_mae_before': metrics.get('dbp_mae_before', 0),
            'dbp_mae_after': metrics.get('dbp_mae_after', 0),
            'improvement_dbp': metrics.get('improvement_dbp', 0),
            'message': f'{active_model_desc.title()} calibration active'
        }

    def calibrate_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply calibration to all predictions in a DataFrame.

        Args:
            df: DataFrame with predictions

        Returns:
            DataFrame with calibrated_sbp and calibrated_dbp columns added
        """
        # Make a copy to avoid modifying original
        df = df.copy()

        if not self.is_fitted:
            df['calibrated_sbp'] = df['sbp_mean']
            df['calibrated_dbp'] = df['dbp_mean']
            df['calibration_model'] = 'none'
            return df

        calibrated_results = []

        status = self.get_status()
        active_model = status.get('active_model', 'none')

        for idx, row in df.iterrows():
            cal_sbp, cal_dbp = self.predict(row['sbp_mean'], row['dbp_mean'], None)
            calibrated_results.append({
                'calibrated_sbp': cal_sbp,
                'calibrated_dbp': cal_dbp,
                'calibration_model': active_model
            })

        # Add calibrated columns - drop existing ones first to avoid duplicates
        for col in ['calibrated_sbp', 'calibrated_dbp', 'calibration_model']:
            if col in df.columns:
                df = df.drop(columns=[col])

        cal_df = pd.DataFrame(calibrated_results, index=df.index)
        df = pd.concat([df, cal_df], axis=1)

        return df

    def clear(self):
        """Clear current calibration model."""
        import os
        import shutil

        # Delete all calibration files in the directory
        if self.model_dir.exists():
            for item in self.model_dir.iterdir():
                try:
                    if item.is_file():
                        os.chmod(item, 0o777)
                        item.unlink()
                    elif item.is_dir():
                        shutil.rmtree(item)
                except Exception as e:
                    print(f"Warning: Could not delete {item}: {e}")

        # Also clean up old hybrid_calibration directory if it exists
        old_dir = self.model_dir / "hybrid_calibration"
        if old_dir.exists():
            try:
                for file_path in old_dir.glob('*'):
                    try:
                        if file_path.is_file():
                            os.chmod(file_path, 0o777)
                            file_path.unlink()
                    except:
                        pass
                old_dir.rmdir()
            except:
                pass

        self.linear_cal = LinearCalibration()
        self.is_fitted = False
