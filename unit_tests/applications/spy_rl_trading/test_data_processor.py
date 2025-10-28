"""Unit tests for SPY Data Processor.

Tests cover:
    - Data download (OHLCV columns, date range, non-null values)
    - Data cleaning (NaN removal, gap detection, outlier flagging)
    - Technical indicators (computation, column count)
    - VIX integration
    - Array conversion (shapes, types)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from finrl.applications.spy_rl_trading.data_processor import SPYDataProcessor
from finrl.config import SPY_INDICATORS


class TestSPYDataProcessor:
    """Test suite for SPYDataProcessor."""

    @pytest.fixture
    def processor(self):
        """Create SPYDataProcessor instance."""
        return SPYDataProcessor()

    @pytest.fixture
    def sample_data(self):
        """Create sample SPY data for testing."""
        dates = pd.date_range("2020-01-01", periods=100, freq="B")  # Business days
        data = {
            "date": dates,
            "open": np.random.uniform(300, 400, 100),
            "high": np.random.uniform(350, 450, 100),
            "low": np.random.uniform(250, 350, 100),
            "close": np.random.uniform(300, 400, 100),
            "volume": np.random.randint(1e6, 1e9, 100),
            "tic": ["SPY"] * 100,
        }
        df = pd.DataFrame(data)

        # Ensure OHLC consistency
        df["high"] = df[["open", "high", "close"]].max(axis=1)
        df["low"] = df[["open", "low", "close"]].min(axis=1)

        return df

    def test_download_data_columns(self, processor):
        """Test download_data returns correct columns."""
        # Note: This test requires internet connection
        # Use small date range to minimize download time
        df = processor.download_data(
            start_date="2024-01-01",
            end_date="2024-01-31",
            ticker_list=["SPY"],
            time_interval="1D",
        )

        # Check required columns exist
        required_cols = ["date", "open", "high", "low", "close", "volume", "tic"]
        for col in required_cols:
            assert col in df.columns, f"Missing column: {col}"

        # Check data types
        assert df["date"].dtype == "datetime64[ns]"
        assert df["tic"].iloc[0] == "SPY"

    def test_download_data_non_null(self, processor):
        """Test download_data returns non-null OHLCV values."""
        df = processor.download_data(
            start_date="2024-01-01", end_date="2024-01-31", ticker_list=["SPY"]
        )

        # Check no NaN in critical columns
        assert df["open"].notna().all()
        assert df["high"].notna().all()
        assert df["low"].notna().all()
        assert df["close"].notna().all()
        assert df["volume"].notna().all()

    def test_clean_data_removes_nan(self, processor, sample_data):
        """Test clean_data removes NaN rows."""
        # Introduce NaN values
        sample_data.loc[5, "close"] = np.nan
        sample_data.loc[10, "open"] = np.nan

        df_clean = processor.clean_data(sample_data)

        # Check NaN rows removed
        assert df_clean["close"].notna().all()
        assert df_clean["open"].notna().all()
        assert len(df_clean) < len(sample_data)

    def test_clean_data_completeness_threshold(self, processor):
        """Test clean_data raises error if completeness <99%."""
        # Create sparse data (50% complete)
        dates = pd.date_range("2020-01-01", periods=50, freq="2B")  # Every 2 days
        sparse_data = pd.DataFrame(
            {
                "date": dates,
                "open": [300] * 50,
                "high": [310] * 50,
                "low": [290] * 50,
                "close": [305] * 50,
                "volume": [1e6] * 50,
                "tic": ["SPY"] * 50,
            }
        )

        with pytest.raises(ValueError, match="Data completeness too low"):
            processor.clean_data(sparse_data)

    def test_clean_data_flags_outliers(self, processor, sample_data):
        """Test clean_data flags outliers (>5σ returns)."""
        # Introduce outlier: 20% drop in one day
        sample_data.loc[50, "close"] = sample_data.loc[49, "close"] * 0.8

        # Capture print output
        import io
        import sys

        captured_output = io.StringIO()
        sys.stdout = captured_output

        df_clean = processor.clean_data(sample_data)

        sys.stdout = sys.__stdout__
        output = captured_output.getvalue()

        # Check warning printed
        assert "WARNING" in output
        assert "outlier" in output.lower()

    def test_clean_data_validates_ohlc(self, processor, sample_data):
        """Test clean_data validates OHLC consistency."""
        # Introduce inconsistency: close > high
        sample_data.loc[20, "close"] = sample_data.loc[20, "high"] + 10

        with pytest.raises(ValueError, match="OHLC consistency violation"):
            processor.clean_data(sample_data)

    def test_add_technical_indicators(self, processor, sample_data):
        """Test add_technical_indicator computes indicators correctly."""
        indicators = ["macd", "rsi_30", "boll_ub", "boll_lb"]

        df_with_indicators = processor.add_technical_indicator(
            sample_data, indicators
        )

        # Check new columns added
        for indicator in indicators:
            assert indicator in df_with_indicators.columns

        # Check indicator values are numeric
        assert df_with_indicators["macd"].dtype in [np.float64, np.float32]
        assert df_with_indicators["rsi_30"].dtype in [np.float64, np.float32]

    def test_add_technical_indicators_count(self, processor, sample_data):
        """Test correct number of indicators added."""
        indicators = SPY_INDICATORS[:-1]  # Exclude VIX (added separately)
        original_cols = len(sample_data.columns)

        df_with_indicators = processor.add_technical_indicator(
            sample_data, indicators
        )

        # Check column count increased
        assert len(df_with_indicators.columns) == original_cols + len(indicators)

    def test_add_vix(self, processor):
        """Test add_vix downloads and merges VIX data."""
        # Download small SPY dataset
        df = processor.download_data(
            start_date="2024-01-01", end_date="2024-01-31", ticker_list=["SPY"]
        )

        df_with_vix = processor.add_vix(df)

        # Check VIX column exists
        assert "vix" in df_with_vix.columns

        # Check VIX values are non-negative
        assert (df_with_vix["vix"] >= 0).all()

    def test_df_to_array_shapes(self, processor, sample_data):
        """Test df_to_array returns correct array shapes."""
        # Add indicators first
        indicators = ["macd", "rsi_30", "boll_ub", "boll_lb"]
        df_with_indicators = processor.add_technical_indicator(
            sample_data, indicators
        )

        # Add turbulence
        df_with_turb = processor.calculate_turbulence(df_with_indicators)

        price_array, tech_array, turbulence_array = processor.df_to_array(
            df_with_turb, indicators, if_vix=False
        )

        # Check shapes
        assert price_array.shape == (100, 1)
        assert tech_array.shape == (100, 4)  # 4 indicators
        assert turbulence_array.shape == (100,)

    def test_df_to_array_types(self, processor, sample_data):
        """Test df_to_array returns numpy arrays."""
        indicators = ["macd", "rsi_30"]
        df_with_indicators = processor.add_technical_indicator(
            sample_data, indicators
        )
        df_with_turb = processor.calculate_turbulence(df_with_indicators)

        price_array, tech_array, turbulence_array = processor.df_to_array(
            df_with_turb, indicators, if_vix=False
        )

        # Check types
        assert isinstance(price_array, np.ndarray)
        assert isinstance(tech_array, np.ndarray)
        assert isinstance(turbulence_array, np.ndarray)

    def test_calculate_turbulence(self, processor, sample_data):
        """Test calculate_turbulence computes rolling std correctly."""
        df_with_turb = processor.calculate_turbulence(sample_data, window=20)

        # Check turbulence column exists
        assert "turbulence" in df_with_turb.columns

        # Check turbulence is non-negative (std cannot be negative)
        assert (df_with_turb["turbulence"] >= 0).all()

        # Check first 19 values are 0 (insufficient window)
        assert (df_with_turb["turbulence"].iloc[:19] == 0).all()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
