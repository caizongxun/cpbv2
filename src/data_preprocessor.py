import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.decomposition import PCA
from typing import Tuple, List, Dict
import logging
import pickle

logger = logging.getLogger(__name__)

class DataPreprocessor:
    """
    Data preprocessing pipeline for time-series LSTM training.
    Includes normalization, feature selection, and sequence creation.
    """
    
    def __init__(self, df: pd.DataFrame, lookback_period: int = 60):
        self.df = df.copy()
        self.lookback_period = lookback_period
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.pca = None
        self.feature_names = []
        
    def remove_nans(self) -> pd.DataFrame:
        """Remove NaN values from DataFrame."""
        initial_len = len(self.df)
        self.df = self.df.dropna()
        logger.info(f"Removed {initial_len - len(self.df)} NaN rows")
        return self.df
    
    def select_features(
        self,
        feature_cols: List[str],
        n_components: int = 30,
        use_pca: bool = True
    ) -> pd.DataFrame:
        """Select features using correlation and PCA.
        
        Args:
            feature_cols: List of feature column names
            n_components: Number of components for PCA
            use_pca: Whether to use PCA for dimensionality reduction
            
        Returns:
            DataFrame with selected features
        """
        logger.info(f"Selecting features from {len(feature_cols)} candidates...")
        
        # Remove highly correlated features
        feature_data = self.df[feature_cols].copy()
        correlation_matrix = feature_data.corr().abs()
        
        # Find highly correlated pairs
        to_drop = set()
        for i in range(len(correlation_matrix.columns)):
            for j in range(i + 1, len(correlation_matrix.columns)):
                if abs(correlation_matrix.iloc[i, j]) > 0.95:
                    col_to_drop = correlation_matrix.columns[j]
                    to_drop.add(col_to_drop)
        
        feature_cols = [col for col in feature_cols if col not in to_drop]
        logger.info(f"After correlation filtering: {len(feature_cols)} features")
        
        # PCA for dimensionality reduction
        if use_pca and len(feature_cols) > n_components:
            feature_data = self.df[feature_cols].copy()
            self.pca = PCA(n_components=min(n_components, len(feature_cols)))
            feature_data_pca = self.pca.fit_transform(feature_data)
            
            # Replace with PCA components
            pca_df = pd.DataFrame(
                feature_data_pca,
                columns=[f'pca_{i}' for i in range(self.pca.n_components_)],
                index=self.df.index
            )
            self.df = pd.concat([self.df[['timestamp', 'symbol', 'interval']], pca_df], axis=1)
            self.feature_names = list(pca_df.columns)
            
            logger.info(f"PCA reduced to {self.pca.n_components_} components")
            logger.info(f"Explained variance: {sum(self.pca.explained_variance_ratio_):.4f}")
        else:
            self.feature_names = feature_cols
            self.df = self.df[['timestamp', 'symbol', 'interval'] + feature_cols]
        
        return self.df
    
    def normalize_features(self, fit: bool = True) -> pd.DataFrame:
        """Normalize features to [0, 1] range.
        
        Args:
            fit: If True, fit the scaler; if False, use existing scaler
            
        Returns:
            DataFrame with normalized features
        """
        feature_cols = self.feature_names
        
        if fit:
            self.df[feature_cols] = self.scaler.fit_transform(self.df[feature_cols])
            logger.info("Fitted MinMaxScaler to data")
        else:
            self.df[feature_cols] = self.scaler.transform(self.df[feature_cols])
            logger.info("Applied existing MinMaxScaler")
        
        return self.df
    
    def create_sequences(
        self,
        lookback: Optional[int] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Create time-series sequences for LSTM.
        
        Args:
            lookback: Number of previous timesteps to use
            
        Returns:
            Tuple of (X, y) arrays
        """
        if lookback is None:
            lookback = self.lookback_period
        
        feature_cols = self.feature_names
        data = self.df[feature_cols].values
        
        X, y = [], []
        for i in range(lookback, len(data)):
            X.append(data[i - lookback:i])
            # Target: next 7 days average price change (dummy, will be replaced with actual target)
            y.append(data[i, 0])  # Simple close price prediction
        
        X = np.array(X)
        y = np.array(y).reshape(-1, 1)
        
        logger.info(f"Created {len(X)} sequences of length {lookback}")
        logger.info(f"X shape: {X.shape}, y shape: {y.shape}")
        
        return X, y
    
    def split_train_val_test(
        self,
        X: np.ndarray,
        y: np.ndarray,
        train_ratio: float = 0.70,
        val_ratio: float = 0.15
    ) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
        """Split data into train/val/test (time-series aware).
        
        Args:
            X: Feature array
            y: Target array
            train_ratio: Proportion of training data
            val_ratio: Proportion of validation data
            
        Returns:
            Dictionary with train/val/test splits
        """
        n = len(X)
        train_idx = int(n * train_ratio)
        val_idx = int(n * (train_ratio + val_ratio))
        
        X_train = X[:train_idx]
        y_train = y[:train_idx]
        
        X_val = X[train_idx:val_idx]
        y_val = y[train_idx:val_idx]
        
        X_test = X[val_idx:]
        y_test = y[val_idx:]
        
        logger.info(f"Train: {X_train.shape[0]}, Val: {X_val.shape[0]}, Test: {X_test.shape[0]}")
        
        return {
            'X_train': X_train, 'y_train': y_train,
            'X_val': X_val, 'y_val': y_val,
            'X_test': X_test, 'y_test': y_test
        }
    
    def save_scaler(self, filepath: str):
        """Save scaler for inference."""
        with open(filepath, 'wb') as f:
            pickle.dump(self.scaler, f)
        logger.info(f"Scaler saved to {filepath}")
    
    def load_scaler(self, filepath: str):
        """Load scaler for inference."""
        with open(filepath, 'rb') as f:
            self.scaler = pickle.load(f)
        logger.info(f"Scaler loaded from {filepath}")
    
    def save_pca(self, filepath: str):
        """Save PCA model for inference."""
        if self.pca is not None:
            with open(filepath, 'wb') as f:
                pickle.dump(self.pca, f)
            logger.info(f"PCA model saved to {filepath}")
    
    def load_pca(self, filepath: str):
        """Load PCA model for inference."""
        with open(filepath, 'rb') as f:
            self.pca = pickle.load(f)
        logger.info(f"PCA model loaded from {filepath}")


from typing import Optional
