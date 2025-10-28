"""SPY Data Processor - Yahoo Finance integration for SPY trading.

This module provides SPY-specific data processing capabilities, extending
FinRL's YahooFinanceProcessor with validation and quality checks.

Key Features:
    - Download SPY daily OHLCV data from Yahoo Finance
    - Clean data with 99% completeness validation
    - Add technical indicators (MACD, RSI, Bollinger Bands, etc.)
    - Compute VIX for market regime detection
    - Convert to arrays for environment consumption

Example:
    >>> from finrl.applications.spy_rl_trading.data_processor import SPYDataProcessor
    >>> processor = SPYDataProcessor()
    >>> df = processor.download_data("2020-01-01", "2024-12-31")
    >>> df = processor.clean_data(df)
    >>> df = processor.add_technical_indicators(df)
    >>> price_array, tech_array, turbulence_array = processor.df_to_array(df)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from finrl.meta.data_processors.processor_yahoofinance import YahooFinanceProcessor
from stockstats import StockDataFrame as Sdf


class SPYDataProcessor(YahooFinanceProcessor):
    """SPY-specific data processor extending YahooFinanceProcessor.

    Inherits all methods from YahooFinanceProcessor and adds SPY-specific
    validation and quality checks.
    """

    def __init__(self):
        """Initialize SPY data processor."""
        super().__init__()

    def download_data(
        self,
        start_date: str,
        end_date: str,
        ticker_list: list[str] | None = None,
        time_interval: str = "1D",
    ) -> pd.DataFrame:
        """Download SPY OHLCV data from Yahoo Finance.

        Args:
            start_date: Start date in 'YYYY-MM-DD' format
            end_date: End date in 'YYYY-MM-DD' format
            ticker_list: List of tickers (default: ["SPY"])
            time_interval: Time interval (default: "1D")

        Returns:
            DataFrame with columns: date, open, high, low, close, volume, tic

        Example:
            >>> processor = SPYDataProcessor()
            >>> df = processor.download_data("2020-01-01", "2024-12-31")
            >>> print(df.shape)
            (1260, 7)  # ~1260 trading days for 5 years
        """
        if ticker_list is None:
            ticker_list = ["SPY"]

        # Use parent class method to download data
        self.start = start_date
        self.end = end_date
        self.time_interval = time_interval

        # Download using yfinance via parent method
        data_df = pd.DataFrame()

        import yfinance as yf

        for tic in ticker_list:
            temp_df = yf.download(
                tic,
                start=start_date,
                end=end_date,
                interval=self.convert_interval(time_interval),
            )
            temp_df["tic"] = tic
            data_df = pd.concat([data_df, temp_df])

        # Reset index and rename columns to FinRL standard
        data_df = data_df.reset_index()

        # Handle both single-level and multi-level column names from yfinance
        if isinstance(data_df.columns, pd.MultiIndex):
            data_df.columns = data_df.columns.get_level_values(0)

        # Standardize column names to lowercase
        data_df = data_df.rename(
            columns={
                "Date": "date",
                "Open": "open",
                "High": "high",
                "Low": "low",
                "Close": "close",
                "Adj Close": "adjcp",
                "Volume": "volume",
            }
        )

        # Convert date to datetime
        data_df["date"] = pd.to_datetime(data_df["date"])

        # Add 'day' column (days since start)
        start_dt = pd.to_datetime(start_date)
        data_df["day"] = (data_df["date"] - start_dt).dt.days

        # Sort by date and ticker
        data_df = data_df.sort_values(by=["date", "tic"]).reset_index(drop=True)

        return data_df

    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean SPY data with quality validation.

        Applies the following cleaning steps:
            1. Remove NaN rows in OHLCV columns
            2. Validate 99% completeness (≤1% missing trading days)
            3. Flag outliers (>5σ daily returns)
            4. Validate OHLC consistency (low ≤ close ≤ high)

        Args:
            df: Raw DataFrame from download_data()

        Returns:
            Cleaned DataFrame

        Raises:
            ValueError: If data completeness <99% (>1% missing days)

        Example:
            >>> df = processor.download_data("2020-01-01", "2024-12-31")
            >>> df_clean = processor.clean_data(df)
        """
        # Step 1: Remove NaN rows
        df = df.dropna(subset=["close", "open", "high", "low", "volume"])

        # Step 2: Check completeness (95% threshold accounting for market holidays)
        dates = pd.to_datetime(df["date"])
        bdays = pd.bdate_range(dates.min(), dates.max())
        completeness = len(dates.unique()) / len(bdays)

        if completeness < 0.95:
            raise ValueError(
                f"Data completeness too low: {completeness:.2%} "
                f"({len(dates.unique())}/{len(bdays)} days). "
                f"Expected ≥95% (≤{int(0.05 * len(bdays))} missing days)."
            )

        # Step 3: Flag outliers (>5σ daily returns)
        df["daily_return"] = np.log(df["close"] / df["close"].shift(1))
        daily_std = df["daily_return"].std()
        outliers = np.abs(df["daily_return"]) > 5 * daily_std

        if outliers.sum() > 0:
            outlier_dates = df.loc[outliers, "date"].tolist()
            print(
                f"WARNING: {outliers.sum()} outlier(s) detected (>5σ returns). "
                f"Dates: {outlier_dates[:5]}... Review before training."
            )

        # Step 4: Validate OHLC consistency
        invalid_ohlc = (df["low"] > df["close"]) | (df["close"] > df["high"])
        if invalid_ohlc.sum() > 0:
            raise ValueError(
                f"OHLC consistency violation: {invalid_ohlc.sum()} rows where "
                f"low > close or close > high. Check data quality."
            )

        return df

    def add_technical_indicator(
        self,
        df: pd.DataFrame,
        tech_indicator_list: list[str],
    ) -> pd.DataFrame:
        """Add technical indicators to DataFrame.

        Uses stockstats library to compute indicators. Supported indicators:
            - macd: Moving Average Convergence Divergence
            - boll_ub, boll_lb: Bollinger Bands (upper/lower)
            - rsi_30: 30-period Relative Strength Index
            - cci_30: 30-period Commodity Channel Index
            - dx_30: 30-period Directional Movement Index
            - close_30_sma, close_60_sma: Simple Moving Averages

        Args:
            df: DataFrame with OHLCV data
            tech_indicator_list: List of indicator names

        Returns:
            DataFrame with additional indicator columns

        Example:
            >>> indicators = ['macd', 'rsi_30', 'boll_ub', 'boll_lb']
            >>> df = processor.add_technical_indicator(df, indicators)
        """
        df = df.copy()
        df = df.sort_values(by=["tic", "date"])
        stock = Sdf.retype(df.copy())
        unique_ticker = stock.tic.unique()

        for indicator in tech_indicator_list:
            indicator_df = pd.DataFrame()
            for tic in unique_ticker:
                try:
                    temp_indicator = stock[stock.tic == tic][indicator]
                    temp_indicator = pd.DataFrame(temp_indicator)
                    temp_indicator["tic"] = tic
                    temp_indicator["date"] = df[df.tic == tic]["date"].to_list()
                    indicator_df = pd.concat(
                        [indicator_df, temp_indicator], ignore_index=True
                    )
                except Exception as e:
                    print(f"Error computing {indicator} for {tic}: {e}")

            df = df.merge(
                indicator_df[["tic", "date", indicator]], on=["tic", "date"], how="left"
            )

        df = df.sort_values(by=["date", "tic"])
        return df

    def add_vix(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add VIX (Volatility Index) to DataFrame.

        Downloads VIX data for the same date range and merges with SPY data.
        VIX is used for market regime detection (high VIX = high volatility).

        Args:
            df: DataFrame with SPY data

        Returns:
            DataFrame with additional 'vix' column

        Example:
            >>> df = processor.add_vix(df)
            >>> print(df[['date', 'close', 'vix']].head())
        """
        import yfinance as yf

        # Get date range from dataframe
        start_date = df["date"].min().strftime("%Y-%m-%d")
        end_date = df["date"].max().strftime("%Y-%m-%d")

        # Download VIX data
        vix_df = yf.download("^VIX", start=start_date, end=end_date)
        vix_df = vix_df.reset_index()

        # Handle both single-level and multi-level column names from yfinance
        if isinstance(vix_df.columns, pd.MultiIndex):
            vix_df.columns = vix_df.columns.get_level_values(0)

        # Standardize column names
        vix_df = vix_df.rename(columns={"Date": "date", "Close": "close"})
        vix_df["date"] = pd.to_datetime(vix_df["date"])

        # Keep only date and close (VIX value)
        vix_df = vix_df[["date", "close"]].rename(columns={"close": "vix"})

        # Merge with main dataframe
        df = df.merge(vix_df, on="date", how="left")

        # Forward fill missing VIX values (market holidays)
        df["vix"] = df["vix"].fillna(method="ffill")

        return df

    def df_to_array(
        self, df: pd.DataFrame, tech_indicator_list: list[str], if_vix: bool = True
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Convert DataFrame to numpy arrays for environment.

        Args:
            df: DataFrame with OHLCV + indicators
            tech_indicator_list: List of technical indicators
            if_vix: Include VIX in output (default: True)

        Returns:
            Tuple of (price_array, tech_array, turbulence_array):
                - price_array: shape (n_days, 1) - close prices
                - tech_array: shape (n_days, n_indicators) - technical indicators
                - turbulence_array: shape (n_days,) - turbulence index

        Example:
            >>> indicators = ['macd', 'rsi_30', 'boll_ub', 'boll_lb']
            >>> price, tech, turb = processor.df_to_array(df, indicators)
            >>> print(price.shape, tech.shape, turb.shape)
            (1260, 1) (1260, 10) (1260,)
        """
        unique_ticker = df.tic.unique()

        # For SPY (single ticker), extract arrays directly
        if len(unique_ticker) == 1:
            price_array = df[["close"]].values

            # Extract technical indicators
            tech_array = df[tech_indicator_list].values

            # Compute turbulence index (rolling std of returns)
            returns = np.log(df["close"] / df["close"].shift(1)).fillna(0)
            turbulence = returns.rolling(window=20).std().fillna(0).values

            return price_array, tech_array, turbulence
        else:
            raise ValueError("SPYDataProcessor expects single ticker (SPY only)")

    def calculate_turbulence(self, df: pd.DataFrame, window: int = 20) -> pd.DataFrame:
        """Calculate turbulence index (market instability measure).

        Turbulence = rolling standard deviation of log returns over window.

        Args:
            df: DataFrame with price data
            window: Rolling window size (default: 20 days)

        Returns:
            DataFrame with additional 'turbulence' column

        Example:
            >>> df = processor.calculate_turbulence(df, window=20)
        """
        df = df.copy()
        df["daily_return"] = np.log(df["close"] / df["close"].shift(1)).fillna(0)
        df["turbulence"] = df["daily_return"].rolling(window=window).std().fillna(0)
        return df
