import yfinance as yf
import pandas as pd
from pathlib import Path

# header
# [('Close', 'SHEL'), ('High', 'SHEL'), ('Low', 'SHEL'), ('Open', 'SHEL'), ('Volume', 'SHEL')]

def stock_data(tickers: list | str, p = "6mo"):
    for t in tickers:
        filepath = Path(f"datacapsules/{t+p}.csv")
        df = yf.download([t], period = p)
        df.to_csv(filepath)
