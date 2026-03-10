# Finance-Research

A quantitative research framework that combines **similarity-based market regime retrieval**, **portfolio strategy backtesting**, and an **interpretable explanation layer**.

This project demonstrates how retrieval-based methods can be used to analyze financial market regimes and generate explainable trading signals.

---

# Project Overview

The system builds a research pipeline that:

1. Loads historical market price data  
2. Extracts features and constructs a similarity index  
3. Retrieves historically similar market regimes  
4. Generates trading signals based on retrieved regimes  
5. Constructs portfolio weights  
6. Runs backtests to evaluate strategy performance  
7. Produces interpretable explanations for trading decisions

The architecture integrates **quantitative finance methods** with **retrieval-based reasoning frameworks**.

---

# Pipeline

The research pipeline operates as follows:
Market Data
↓
Feature Engineering
↓
Similarity Retrieval
↓
Signal Generation
↓
Portfolio Construction
↓
Backtesting
↓
Explanation Layer


The similarity retrieval stage identifies historical market regimes that resemble the current market state, enabling the strategy to leverage patterns observed in past data.

---

# Key Components

## Similarity Retrieval

The module in `chunk/` builds a similarity index for market regimes and retrieves historical neighbors for the current market state.

This stage functions similarly to the **retrieval component in retrieval-augmented systems**, where historical market states serve as reference knowledge.

---

## Portfolio Strategy

The `strategy/` module contains the portfolio construction and evaluation logic, including:

- Equal-weight baseline strategy  
- Similarity-based portfolio weighting  
- Portfolio backtesting utilities  
- Volatility targeting adjustments  

Example workflow:
load_prices()
↓
build similarity weights
↓
backtest portfolio


---

## Explanation Layer

The module in `llm/` generates interpretable explanations for trading signals.

The explanations are deterministic and built from retrieved evidence such as:

- predicted return quantiles
- neighbor similarity statistics
- drawdown statistics
- decision thresholds

This ensures the trading decisions remain **transparent and reproducible**.

The repository also includes experiments with LLM-generated explanations in `LLM_reason.ipynb`.

---


The script will:

1. Load market data  
2. run the baseline equal-weight strategy  
3. run the similarity-based strategy  
4. compute backtest performance metrics  
5. generate interpretable trading signal explanations  

---

# Example Output

Running the pipeline produces:

- portfolio performance metrics  
- retrieved similar market regimes  
- trading signal explanations  
- risk statistics such as drawdown  

---


