# News Sentiment & Stock Trading Strategy

Pipeline that scores Chinese financial news sentiment and uses it (weighted by news popularity) to build and backtest a stock-selection strategy against the CSI 300 index (2018–2021).

| Step | Folder | Description |
|---|---|---|
| 1 | `1. news_data_combination` | Merge raw news data |
| 2 | `2. EDA` | Popular stocks, news sources, correlation, topic modeling |
| 3 | `3. sentiment_analysis` | LSTM model and classical ML models for sentiment scoring |
| 4 | `4. popularity` | News popularity metric |
| 5 | `5. news+mkt_data_combination` | Join news sentiment with market data |
| 6 | `6. mkt_index_data` | Market index data |
| 7 | `7. trading_strategy` | Strategy export and backtesting (Sharpe, Calmar, max drawdown) |
| 8 | `8. data_colletion` | Link to full dataset (Dropbox) |

## Notes
- Data: 2017–2021 Chinese financial news and A-share market data from the CSMAR database, provided by the course instructor.
- The full dataset is too large for the repo; see `8. data_colletion/dropbox_link.txt`.
- EDA charts use the SimHei font for Chinese labels (not included); place `SimHei.ttf` in `2. EDA/` to run them.
- File paths in the scripts are relative and may need adjusting.

---

<sub>Group project for MSBA7012, Master of Science in Business Analytics, The University of Hong Kong (February 2023): <i>A Profitable Trading Strategy Through Sentiment Analysis on Chinese Financial News</i>.<br>
Contributors (Group 1): Cai Shida, Chen Xiaojie, Huang Huiling, Liu Yiwei, Lu Jinnuo, Wang Manlin, Xu Chenyun, Ying Zitao.</sub>
