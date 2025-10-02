# Quantitative Trading System

A sophisticated algorithmic trading framework implementing machine learning-based signal generation and walk-forward optimization for forex markets. The system leverages advanced feature engineering, purged cross-validation, and ensemble methods to generate robust trading strategies.

## Overview

This platform provides an end-to-end quantitative trading solution that combines technical analysis, sentiment features, and machine learning to generate directional forecasts in forex markets. The framework implements key concepts from modern quantitative finance, including fractional differentiation, meta-labeling, and walk-forward validation.

## Key Features

### Machine Learning Pipeline
- **Purged K-Fold Cross-Validation**: Prevents information leakage in time-series data
- **Sample Weighting**: Time-decay and uniqueness-based weighting for training samples
- **Meta-Labeling Architecture**: Secondary model for bet sizing and position management
- **Ensemble Methods**: Bagging with custom pipeline integration

### Feature Engineering
- **Triple-Barrier Labeling**: Dynamic profit targets and stop losses based on volatility
- **CUSUM Filter**: Identifies significant directional moves for event-driven sampling
- **Technical Indicators**: Moving averages, Bollinger Bands, support/resistance detection
- **Fractional Differentiation**: Achieves stationarity while preserving memory

### Risk Management
- **Walk-Forward Optimization**: Prevents overfitting through expanding window validation
- **Dynamic Position Sizing**: Risk-adjusted sizing based on model confidence
- **Volatility-Based Stops**: Adaptive stop-loss and take-profit levels

### Live Trading Integration
- Interactive Brokers API integration for paper and live trading
- Real-time data processing and signal generation
- Automated order execution with bracket orders

## Architecture

```
QS_algo/
├── main.py                 # Entry point and orchestration
├── data_preparation.py     # Data loading, resampling, and feature utilities
├── primary_model.py        # Technical feature generation
├── model_training.py       # ML training pipeline with PurgedKFold
├── walkforward.py          # Walk-forward backtesting engine
├── bet_sizing.py           # Position sizing and meta-labeling
├── trading.py              # Live/paper trading execution
├── ensemble.py             # Ensemble model components
├── polygon.py              # Data provider integration (Polygon.io)
├── tiingo.py               # Data provider integration (Tiingo)
├── scraping.py             # Alternative data collection
└── config.json             # System configuration
```

## Core Methodology

### 1. Data Preparation
The system processes high-frequency forex data (1-minute bars) and applies intelligent resampling to reduce noise while preserving information content. Market hours filtering ensures only active trading periods are included.

### 2. Feature Generation
Features span multiple categories:
- **Technical**: Moving averages, Bollinger Bands, support/resistance levels
- **Sentiment**: News headlines, retail sentiment indicators
- **Microstructure**: Volume-based bars, tick imbalance

### 3. Label Generation
Triple-barrier method creates labels by:
- Setting dynamic profit targets based on volatility
- Applying symmetric or asymmetric barriers
- Identifying which barrier is touched first
- Filtering labels by minimum profit threshold

### 4. Sample Weighting
Implements two weighting schemes:
- **Uniqueness**: Reduces weight of overlapping samples
- **Time Decay**: Recent samples receive higher weights

### 5. Model Training
Random Forest classifier with:
- Purged K-Fold cross-validation (eliminates leakage)
- Embargo period between folds
- Grid/randomized hyperparameter search
- Bagging for variance reduction

### 6. Walk-Forward Testing
Expanding window approach:
- Train on historical data
- Validate on out-of-sample period
- Re-train with updated data
- Ensures model adapts to regime changes

## Installation

```bash
# Clone the repository
git clone <repository-url>
cd QS_algo

# Install dependencies
pip install -r requirements.txt
```

## Configuration

Configure your trading parameters in the main execution block of [main.py](main.py):

```python
params = {
    'userID': '1',
    'name': 'Strategy Name',
    'strategy_objective': 'risk_minimization',
    'asset_selection': 'forex',
    'asset_name': ['EURUSD'],
    'trading_frequency': 'minutes',
    'position_holding': 'short_term',
    'risk_tolerance': 'low',

    'technical_feature': ['moving_average'],
    'sentiment_feature': ['news_headlines', 'retail_sentiment'],

    'entry_long_signal': ['price_above_sma', 'retracement'],
    'entry_short_signal': ['price_below_sma', 'retracement'],

    'initial_investment': 100000,
    'ml_model': 'random_forest',
    'action': 'create'  # 'create', 'update', 'paper', 'live'
}
```

## Usage

### Backtesting Mode
```python
params['action'] = 'create'
main(params)
```

This will:
1. Load and prepare historical data
2. Generate features and labels
3. Train the ML model with purged cross-validation
4. Execute walk-forward backtest
5. Return performance metrics and equity curve

### Paper Trading Mode
```python
params['action'] = 'paper'
main(params)
```

Connects to Interactive Brokers paper trading account and executes trades in real-time.

### Live Trading Mode
```python
params['action'] = 'live'
main(params)
```

⚠️ **WARNING**: Only use after thorough testing. Real capital is at risk.

## Performance Metrics

The system reports comprehensive statistics:
- **Return [%]**: Total percentage return
- **Volatility (Ann.) [%]**: Annualized volatility
- **Sharpe Ratio**: Risk-adjusted returns
- **Max Drawdown [%]**: Largest peak-to-trough decline
- **Equity Curve**: Time-series of portfolio value
- **Drawdown Series**: Real-time drawdown tracking

## Key Dependencies

- **scikit-learn**: Machine learning models and validation
- **pandas/numpy**: Data manipulation and numerical computing
- **ib_insync**: Interactive Brokers API integration
- **backtesting.py**: Vectorized backtesting engine
- **ta**: Technical analysis indicators
- **scipy/statsmodels**: Statistical analysis and time-series
- **arch**: GARCH modeling for volatility
- **hurst/nolds**: Hurst exponent and fractal analysis

## Research Background

This implementation draws from advances in quantitative finance research:
- Lopez de Prado's work on purged cross-validation and meta-labeling
- Triple-barrier method for asymmetric bet structures
- Fractional differentiation for stationarity without memory loss
- Sample weighting for non-IID financial data

## Limitations & Future Work

**Current Limitations:**
- Limited to single-asset strategies
- Forex-specific implementation
- No multi-period optimization
- Basic execution model (no slippage modeling)

**Planned Enhancements:**
- Multi-asset portfolio optimization
- Alternative data integration (satellite, credit card, etc.)
- Deep learning models (LSTM, Transformers)
- Transaction cost analysis and optimization
- Regime detection and switching

## Risk Disclaimer

This software is for educational and research purposes. Algorithmic trading involves substantial risk of loss. Past performance does not guarantee future results. The authors assume no liability for financial losses incurred through use of this system.

## Contributing

Contributions are welcome. Please follow these guidelines:
- Fork the repository
- Create a feature branch
- Add tests for new functionality
- Ensure all tests pass
- Submit a pull request with detailed description

## Acknowledgments

This project implements methodologies from:
- **Advances in Financial Machine Learning** by Marcos Lopez de Prado
- **Machine Learning for Algorithmic Trading** by Stefan Jansen
- Academic research in quantitative finance and market microstructure

## License

MIT License - see LICENSE file for details

## Author

Pablo Pedrosa

---

**Note**: This is a research and educational project. Always conduct thorough due diligence before deploying any algorithmic trading system with real capital.
