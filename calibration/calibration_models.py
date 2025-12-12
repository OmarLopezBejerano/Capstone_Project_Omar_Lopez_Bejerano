import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from typing import Dict, Tuple, Optional
import json
from pathlib import Path


class LinearCalibration:

    def __init__(self):
        self.sbp_model = LinearRegression()
        self.dbp_model = LinearRegression()
        self.is_fitted = False
        self.n_samples = 0
        self.metrics = {}

    def fit(self, df: pd.DataFrame) -> Dict:
        # Filter valid data
        valid_data = df[(df['cuff_sbp'].notna()) & (df['cuff_dbp'].notna())].copy()

        if len(valid_data) < 3:
            return {
                'success': False,
                'error': f'Need at least 3 recordings with cuff measurements. Found: {len(valid_data)}'
            }

        self.n_samples = len(valid_data)

        # Prepare features (predicted BP) and targets (cuff BP)
        X_sbp = valid_data['sbp_mean'].values.reshape(-1, 1)
        y_sbp = valid_data['cuff_sbp'].values

        X_dbp = valid_data['dbp_mean'].values.reshape(-1, 1)
        y_dbp = valid_data['cuff_dbp'].values

        # Fit linear models
        self.sbp_model.fit(X_sbp, y_sbp)
        self.dbp_model.fit(X_dbp, y_dbp)

        self.is_fitted = True

        # Calculate metrics
        sbp_pred_calibrated = self.sbp_model.predict(X_sbp)
        dbp_pred_calibrated = self.dbp_model.predict(X_dbp)

        # Before calibration errors
        sbp_error_before = np.abs(valid_data['sbp_mean'].values - y_sbp).mean()
        dbp_error_before = np.abs(valid_data['dbp_mean'].values - y_dbp).mean()

        # After calibration errors
        sbp_error_after = np.abs(sbp_pred_calibrated - y_sbp).mean()
        dbp_error_after = np.abs(dbp_pred_calibrated - y_dbp).mean()

        self.metrics = {
            'success': True,
            'n_samples': self.n_samples,
            'sbp_coef': float(self.sbp_model.coef_[0]),
            'sbp_intercept': float(self.sbp_model.intercept_),
            'dbp_coef': float(self.dbp_model.coef_[0]),
            'dbp_intercept': float(self.dbp_model.intercept_),
            'sbp_mae_before': float(sbp_error_before),
            'sbp_mae_after': float(sbp_error_after),
            'dbp_mae_before': float(dbp_error_before),
            'dbp_mae_after': float(dbp_error_after),
            'improvement_sbp': float(sbp_error_before - sbp_error_after),
            'improvement_dbp': float(dbp_error_before - dbp_error_after)
        }

        return self.metrics

    def predict(self, sbp_pred: float, dbp_pred: float) -> Tuple[float, float]:

        if not self.is_fitted:
            return sbp_pred, dbp_pred

        sbp_cal = self.sbp_model.predict([[sbp_pred]])[0]
        dbp_cal = self.dbp_model.predict([[dbp_pred]])[0]

        return float(sbp_cal), float(dbp_cal)

    def save(self, path: str):
        import os

        data = {
            'sbp_coef': self.sbp_model.coef_[0] if self.is_fitted else None,
            'sbp_intercept': self.sbp_model.intercept_ if self.is_fitted else None,
            'dbp_coef': self.dbp_model.coef_[0] if self.is_fitted else None,
            'dbp_intercept': self.dbp_model.intercept_ if self.is_fitted else None,
            'is_fitted': self.is_fitted,
            'n_samples': self.n_samples,
            'metrics': self.metrics
        }

        if os.path.exists(path):
            try:
                os.chmod(path, 0o644)
            except:
                pass

        with open(path, 'w') as f:
            json.dump(data, f, indent=2)

        # Set permissions after writing
        try:
            os.chmod(path, 0o644)
        except:
            pass

    def load(self, path: str):
        with open(path, 'r') as f:
            data = json.load(f)

        if data['is_fitted']:
            self.sbp_model.coef_ = np.array([data['sbp_coef']])
            self.sbp_model.intercept_ = data['sbp_intercept']
            self.dbp_model.coef_ = np.array([data['dbp_coef']])
            self.dbp_model.intercept_ = data['dbp_intercept']
            self.is_fitted = data['is_fitted']
            self.n_samples = data['n_samples']
            self.metrics = data['metrics']


class PolynomialCalibration:

    def __init__(self, sbp_degree: int = 2, dbp_degree: int = 2):
        self.sbp_degree = sbp_degree
        self.dbp_degree = dbp_degree
        self.sbp_poly = PolynomialFeatures(degree=sbp_degree)
        self.dbp_poly = PolynomialFeatures(degree=dbp_degree)
        self.sbp_model = LinearRegression()
        self.dbp_model = LinearRegression()
        self.is_fitted = False
        self.n_samples = 0
        self.metrics = {}

    def fit(self, df: pd.DataFrame) -> Dict:

        valid_data = df[(df['cuff_sbp'].notna()) & (df['cuff_dbp'].notna())].copy()

        if len(valid_data) < 5:
            return {
                'success': False,
                'error': f'Need at least 5 recordings for polynomial calibration. Found: {len(valid_data)}'
            }

        self.n_samples = len(valid_data)

        # Prepare data
        X_sbp = valid_data['sbp_mean'].values.reshape(-1, 1)
        y_sbp = valid_data['cuff_sbp'].values
        X_dbp = valid_data['dbp_mean'].values.reshape(-1, 1)
        y_dbp = valid_data['cuff_dbp'].values

        # Transform and fit
        X_sbp_poly = self.sbp_poly.fit_transform(X_sbp)
        X_dbp_poly = self.dbp_poly.fit_transform(X_dbp)

        self.sbp_model.fit(X_sbp_poly, y_sbp)
        self.dbp_model.fit(X_dbp_poly, y_dbp)

        self.is_fitted = True

        # Calculate metrics
        sbp_pred_calibrated = self.sbp_model.predict(X_sbp_poly)
        dbp_pred_calibrated = self.dbp_model.predict(X_dbp_poly)

        sbp_error_before = np.abs(valid_data['sbp_mean'].values - y_sbp).mean()
        dbp_error_before = np.abs(valid_data['dbp_mean'].values - y_dbp).mean()
        sbp_error_after = np.abs(sbp_pred_calibrated - y_sbp).mean()
        dbp_error_after = np.abs(dbp_pred_calibrated - y_dbp).mean()

        self.metrics = {
            'success': True,
            'n_samples': self.n_samples,
            'sbp_degree': self.sbp_degree,
            'dbp_degree': self.dbp_degree,
            'sbp_mae_before': float(sbp_error_before),
            'sbp_mae_after': float(sbp_error_after),
            'dbp_mae_before': float(dbp_error_before),
            'dbp_mae_after': float(dbp_error_after),
            'improvement_sbp': float(sbp_error_before - sbp_error_after),
            'improvement_dbp': float(dbp_error_before - dbp_error_after)
        }

        return self.metrics

    def predict(self, sbp_pred: float, dbp_pred: float) -> Tuple[float, float]:
        """Apply polynomial calibration."""
        if not self.is_fitted:
            return sbp_pred, dbp_pred

        X_sbp_poly = self.sbp_poly.transform([[sbp_pred]])
        X_dbp_poly = self.dbp_poly.transform([[dbp_pred]])

        sbp_cal = self.sbp_model.predict(X_sbp_poly)[0]
        dbp_cal = self.dbp_model.predict(X_dbp_poly)[0]

        return float(sbp_cal), float(dbp_cal)

    def save(self, path: str):
        """Save polynomial calibration model."""
        import os

        data = {
            'sbp_degree': self.sbp_degree,
            'dbp_degree': self.dbp_degree,
            'sbp_coefs': self.sbp_model.coef_.tolist() if self.is_fitted else None,
            'sbp_intercept': float(self.sbp_model.intercept_) if self.is_fitted else None,
            'dbp_coefs': self.dbp_model.coef_.tolist() if self.is_fitted else None,
            'dbp_intercept': float(self.dbp_model.intercept_) if self.is_fitted else None,
            'is_fitted': self.is_fitted,
            'n_samples': self.n_samples,
            'metrics': self.metrics
        }

        if os.path.exists(path):
            try:
                os.chmod(path, 0o644)
            except:
                pass

        with open(path, 'w') as f:
            json.dump(data, f, indent=2)

        try:
            os.chmod(path, 0o644)
        except:
            pass

    def load(self, path: str):
        """Load polynomial calibration model."""
        with open(path, 'r') as f:
            data = json.load(f)

        if data['is_fitted']:
            self.sbp_degree = data['sbp_degree']
            self.dbp_degree = data['dbp_degree']
            self.sbp_poly = PolynomialFeatures(degree=self.sbp_degree)
            self.dbp_poly = PolynomialFeatures(degree=self.dbp_degree)
            # Fit poly features on dummy data to set up transformation
            self.sbp_poly.fit([[0]])
            self.dbp_poly.fit([[0]])

            self.sbp_model.coef_ = np.array(data['sbp_coefs'])
            self.sbp_model.intercept_ = data['sbp_intercept']
            self.dbp_model.coef_ = np.array(data['dbp_coefs'])
            self.dbp_model.intercept_ = data['dbp_intercept']
            self.is_fitted = data['is_fitted']
            self.n_samples = data['n_samples']
            self.metrics = data['metrics']


class HybridCalibration:

    def __init__(self):
        self.linear_cal = LinearCalibration()
        self.poly_cal = None
        self.sbp_model_type = 'linear'  # 'linear' or 'polynomial'
        self.dbp_model_type = 'linear'  # 'linear' or 'polynomial'
        self.n_samples = 0

    def fit(self, df: pd.DataFrame, sbp_degree: int = 1, dbp_degree: int = 1) -> Dict:

        valid_data = df[(df['cuff_sbp'].notna()) & (df['cuff_dbp'].notna())]
        self.n_samples = len(valid_data)

        if self.n_samples < 3:
            return {
                'success': False,
                'error': f'Need at least 3 recordings with cuff measurements for calibration. Found: {self.n_samples}'
            }

        # Always fit linear model as baseline
        linear_result = self.linear_cal.fit(df)
        if not linear_result['success']:
            return linear_result

        # Determine if we need polynomial for either component
        use_poly_sbp = sbp_degree > 1
        use_poly_dbp = dbp_degree > 1

        if use_poly_sbp or use_poly_dbp:
            if self.n_samples < 5:
                return {
                    'success': False,
                    'error': f'Need at least 5 recordings for polynomial calibration. Found: {self.n_samples}'
                }

            self.poly_cal = PolynomialCalibration(sbp_degree=sbp_degree, dbp_degree=dbp_degree)
            poly_result = self.poly_cal.fit(df)

            if not poly_result['success']:
                return poly_result

            self.sbp_model_type = 'polynomial' if use_poly_sbp else 'linear'
            self.dbp_model_type = 'polynomial' if use_poly_dbp else 'linear'
        else:
            self.sbp_model_type = 'linear'
            self.dbp_model_type = 'linear'

        return {
            'success': True,
            'n_samples': self.n_samples,
            'sbp_model_type': self.sbp_model_type,
            'dbp_model_type': self.dbp_model_type,
            'sbp_degree': sbp_degree if use_poly_sbp else 1,
            'dbp_degree': dbp_degree if use_poly_dbp else 1
        }

    def predict(self, sbp_pred: float, dbp_pred: float, record: Optional[Dict] = None) -> Tuple[float, float]:

        # Get SBP calibration
        if self.sbp_model_type == 'polynomial' and self.poly_cal is not None:
            sbp_cal, _ = self.poly_cal.predict(sbp_pred, dbp_pred)
        else:
            sbp_cal, _ = self.linear_cal.predict(sbp_pred, dbp_pred)

        # Get DBP calibration
        if self.dbp_model_type == 'polynomial' and self.poly_cal is not None:
            _, dbp_cal = self.poly_cal.predict(sbp_pred, dbp_pred)
        else:
            _, dbp_cal = self.linear_cal.predict(sbp_pred, dbp_pred)

        return sbp_cal, dbp_cal

    def get_info(self) -> Dict:
        info = {
            'n_samples': self.n_samples,
            'sbp_model_type': self.sbp_model_type,
            'dbp_model_type': self.dbp_model_type,
            'is_calibrated': self.linear_cal.is_fitted
        }

        if self.linear_cal.is_fitted:
            info.update({
                'linear_metrics': self.linear_cal.metrics
            })

        if self.poly_cal is not None and self.poly_cal.is_fitted:
            info.update({
                'poly_metrics': self.poly_cal.metrics
            })

        return info

    def save(self, directory: str):
        """Save calibration models."""
        import os

        dir_path = Path(directory)
        dir_path.mkdir(exist_ok=True)

        try:
            os.chmod(dir_path, 0o755)
        except:
            pass

        # Save linear model
        linear_path = dir_path / 'linear_calibration.json'
        self.linear_cal.save(str(linear_path))

        # Save polynomial model if exists
        if self.poly_cal is not None:
            poly_path = dir_path / 'poly_calibration.json'
            self.poly_cal.save(str(poly_path))

        # Save metadata
        metadata_path = dir_path / 'calibration_info.json'
        with open(metadata_path, 'w') as f:
            json.dump({
                'sbp_model_type': self.sbp_model_type,
                'dbp_model_type': self.dbp_model_type,
                'n_samples': self.n_samples,
                'has_poly': self.poly_cal is not None
            }, f, indent=2)

        try:
            os.chmod(metadata_path, 0o644)
        except:
            pass

    def load(self, directory: str) -> bool:
        dir_path = Path(directory)

        if not dir_path.exists():
            return False

        try:
            with open(dir_path / 'calibration_info.json', 'r') as f:
                info = json.load(f)

            self.sbp_model_type = info['sbp_model_type']
            self.dbp_model_type = info['dbp_model_type']
            self.n_samples = info['n_samples']

            # Load linear model
            if (dir_path / 'linear_calibration.json').exists():
                self.linear_cal.load(str(dir_path / 'linear_calibration.json'))

            # Load polynomial model if exists
            if info.get('has_poly', False) and (dir_path / 'poly_calibration.json').exists():
                # Reconstruct polynomial calibration
                with open(dir_path / 'poly_calibration.json', 'r') as f:
                    poly_data = json.load(f)
                self.poly_cal = PolynomialCalibration(
                    sbp_degree=poly_data['sbp_degree'],
                    dbp_degree=poly_data['dbp_degree']
                )
                self.poly_cal.load(str(dir_path / 'poly_calibration.json'))

            return True

        except Exception as e:
            print(f"Error loading calibration: {e}")
            return False
