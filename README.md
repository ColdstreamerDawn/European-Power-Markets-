# European-Power-Markets
XGBoost predictor of Day-Ahead and Imbalance markets across ENTSO-E -> model.py

# 1. Project background and goals
Power generation and market data are incredibly rich and provided to the public via the European ENTSO-E transparency platform (https://transparency.entsoe.eu/, requires account). The platform supports data pulls of a wide range of forecast and actual generation, transmission, consumption and market data at an hourly and sub-hourly scale.

The aim of this project is to create a prediction model trained on historical data to predict key market indicators:
- **Day-Ahead Prices**: c.80-90% of electricity traded in Europe is traded in this market. Participants bid volumes for generation/consumption a day-ahead, with auctions closing the day before power delivery. This market is the most deterministic, with strong correlations to weather and consumption forecasts. **ENTSO-E provides this data.**
- **Intraday Prices**: c.10-15% of electricity traded in Europe is traded after the DA market closes. Regulations vary by country but intraday market auctions can close up to minutes before power delivery (eg. German continuous intraday market). **NOTE: ENTSO-E does not provide this data, so I am unable to work with this market as yet (data is available from paid platforms like EPEX).**
- **Imbalance Prices**: the "backup" or "last resort" market for European electricity markets. This is where any live surplus or shortfalls in actual power delivery are dealt with. This is the most volatile and least deterministic market, and restricted to specialised balancing parties who have the ability to act at short timescales to help the grid (eg. BESS). In line with its volatility, it is also the favourite and often most lucrative market for power traders. **ENTSO-E provides this data, with some countries breaking this further into Long/Short prices for when the market is in over/under-supply**

Self-imposed rules for training: 
- Only use training data which would be available at the time the forecast would be generated. 
- For DA: DA auctions close at 12.00 noon CET, so using sub-daily lagging DA price indicators would be cheating (more on this below). Fair game forecasts such as weather, forecast generation/consumption, previous days' DA prices (eg. price prediction for 14.00 on Day 5 can use the price for 14.00 on Days 1-4, not price at 13.00 on Day 5). 
- For Imbalance: given these are traded live with sub-hourly or live gate closure, lagging indicators and live DA prices can be used. 

# 2. Data sources and training Data
**Data source:**
This project uses data from the ENTSO-E Transparency Platform. To replicate the code you need to create a login and an individual API key.

**Python API client:**
THe ENTSO-E API is difficult to use so I have used a Python client for the API, developed by EnergieID: https://github.com/EnergieID/entsoe-py. The `entsoe-py` package is MIT licensed.

# 3. Training data selection
For the purposes of this current iterations **I have chosen Portugal as my test subject**. It is a relatively isolated European country with only one major interconnection partner (Spain) and a heavily renewables driven power system, making the modelling more deterministic and easier to interpret. The model however is able to take any country given an appropriate country code (eg. France = FR) and appropriate mapping of its interconnect neighbours (eg. for PT this is just ES = Spain). Mappings and country codes provided in the repository **as mappings.txt**.

Note that for all the forecast data pulls I will pull from the target country but also its neighbours. European countries are heavily interconnected and their supply/demand characteristics can influence neighbouring systems (eg. France is a net exporter of nuclear power to Germany).

ENTSO-E training data:
- **Renewables Generation Forecast**: ENTSO-E provides solar and wind forecasts, published D-1 18.00 Brussels time. This is after the DA auction closure, but I believe this is just ENTSO-E delayed in publishing data, with it actually being available for the DA bidders ahead of their 12.00 noon CET deadline https://transparencyplatform.zendesk.com/hc/en-us/articles/16648445340180-Generation-Forecasts-for-Wind-and-Solar-14-1-D
- **Scheduled Cross-Border Exchanges**: Forecast cross-border exchanges of power. This iterates through all of the neighbours for target country. https://transparencyplatform.zendesk.com/hc/en-us/articles/16583853635092-Scheduled-Commercial-Exchanges-12-1-F
- **Load Forecast**: Defined as sum of total generation by a country less balance of import/export exchanges and power absorbed by energy storage. https://transparencyplatform.zendesk.com/hc/en-us/articles/16647979768084-Actual-Total-Load-Day-ahead-Per-Bidding-Zone-6-1-A-6-1-B
- **Market Prices**: Pulls DA prices and Imbalance Long/Short prices. https://transparencyplatform.zendesk.com/hc/en-us/articles/12824886606996-Imbalance-Prices-Total-Imbalance-Volumes-17-1-G-17-1-H & https://transparencyplatform.zendesk.com/hc/en-us/articles/16647234190100-Energy-Prices-12-1-D
- **Imbalance only training data**: Imbalance prices are highly dependent on forecast errors and short-term supply demand swings. After the DA wind and solar forecast is produced, a further intraday forecast exists which I pull. I further pull actual imbalance volumes (lagged 15m).

Synthetic data (created from and within the training data):
- **Lagging indicators**: In general, persistence is a major feature in power price modelling. DA prices tend to be well predicted by the price a day/week before. Imbalance (and Intraday) prices even more so. Therefore for DA I have introduced 24h, 48h, 72h, 1w price lagging indicators (less than 24h would be illegal). For Imbalance we can use immediate lagging indicators at 15m, 1h, 24h and 1w for price and Imbalance volume.
- **General timestamp data**: Power prices have strong correlations to the hour of the day (eg. solar depresses prices at midday, higher consumption in the evening), day of the week (less volatility on weekends), month of the year (solar output lowest in winter, vice-versa for wind). I have created Hour, Day, Month and Weekend variables for the dataset.
- **Residual Load**: European power markets are pay-as-clear markets, where the most expensive type of generation (often coal/gas) sets the price for a given market timestamp. Residual load, defined as load less renewable energy, is a strong indicator of how expensive the price of power will be in a given timestamp. The higher the residual load, the more a country's power will need to be supplied by high-cost fossil sources.
- **Imbalance only synthetic data**: Calculation of the forecasting errors between DA and intraday solar and wind forecasts.

# 4. The XGBoost model
A wise man once told me XGBoost is all you need (https://www.xgblog.ai/p/xgboost-is-all-you-need). The speed and simplicity of gradient-boosted tree models meant these were my first port of call to model power price prediction. Future iterations could consider neural net based models (eg. LSTM, transformers), but I would probably need to secure myself a vastly better rig (GPU, RAM) to make the most of that.

The model takes two inputs: the target test date (eg. 26th May 2026) and how many days to test over (eg. 3 days would test 26th-29th May 2026). All data before the test date is used for training, all data after the test window is discarded.

I used some rather generic hyperparameters for the model after a short grid search. Future iterations will consider tuning the model further.

# 5. Model outputs

In this testing run I will focus on the week beginning 11th May 2026, ending 17th May. The training data set runs from 3rd June 2020 to 2nd June 2026, with data beyond 17th May excluded. The model will train on data from 3rd June 2020 to 10th May 2026 and test on the aforementioned test window.

I have focused on root-mean-squared error (**RMSE**), mean-absolute error (**MAE**) and coefficient of determination (**R^2**) as my error metrics of choice. I have also included a SHAP (SHapley Additive exPlanations) process to interpret the most valuable features of the model. Finally I output some charts to visualise the strength and weaknesses of the models.

# 6. Results: Day-Ahead

**RMSE**: €14.76/MWh

**MAE**: €10.53/MWh

**R^2**: 0.89

The Portugal DA model responds well to the training features, with lagging features and residual load showing strong contributions to model performance. 
<img width="1238" height="470" alt="image" src="https://github.com/user-attachments/assets/bcbe7006-669e-498a-a851-11a3f04d5897" />
<img width="790" height="860" alt="image" src="https://github.com/user-attachments/assets/dab10a10-b8bb-49fa-a851-aac152a8a7df" />

# 7. Results: Imbalance 

**Imbalance Long**:
**RMSE**: €34.20/MWh

**MAE**: €18.81/MWh

**R^2**: 0.59

<img width="1249" height="470" alt="image" src="https://github.com/user-attachments/assets/f192c793-9a0b-43bc-8589-6bf6ed424ab4" />
<img width="790" height="940" alt="image" src="https://github.com/user-attachments/assets/a116d629-ae7c-46a2-8965-69a247c0ad12" />

**Imbalance Short**:
**RMSE**: €31.73/MWh

**MAE**: €18.82/MWh

**R^2**: 0.69

<img width="1249" height="470" alt="image" src="https://github.com/user-attachments/assets/782f6666-fc79-4c09-8976-c74b9d177d57" />
<img width="790" height="940" alt="image" src="https://github.com/user-attachments/assets/427737d8-7a2d-403d-a769-b33f01f33602" />


Given the volatility and non-linear nature of balancing markets, the 0.6-0.7 R^2 compares favourably with expectations. The model shows very strong feature contibution from lagging indicators, with secondary influence from DA price and Imbalance volumes. This aligns with my expectation that balancing markets are hard to calculate deterministically, but still retain strong momentum characteristics in their prices. However, because of the strength of lagging features in the model, the predictions look suspiciously like a lagged version of the previous 15m Balancing Price. This is not a major issue but given the high volatility of balancing prices, making business decisions on the business of lagging indicators may not be optimal.

# 8. Conclusion and next steps

**Conclusions**:
- The models show good performance and speed. The data download took c.12 minutes in my last run, with the training taking <1min. I note this would be fast enough for the model to be retrained every 15 minutes for the Imbalance markets, given the importance of lagging indicators (though the incremental benefit may not be large).
- I anticipate that the major bottlenecks for use cases of ML-based power price predictors are the speed and reliability of European power data. The incremental challenge of adding further country data is one of download speed rather than model architecture: the current Portugal XGBoost works just fine with 200k parameters.
  - There is a further issue of the publication date of ENTSO-E data, which is often a week late with Imbalance metrics. This would be a major limitation of the model to be used live today.
- I am happy with the prediction levels of the models. I anticipate the main usage of short-term power price prediction is focused on power traders and consumers/generators who have significant power market exposure. In that case the main performance indicators of the model are less the RMSE/MAE outputs, rather identification of the hours of the day where min/max peaks occur. I believe the model identifies peaks and troughs well, particularly on DA markets.
  - For example: the first evening peak of the DA prediction set (€130/MWh) comes in far below the actual peak (>€200/MWh). However the model has accurately predicted that the max peak would occur at that time: if I had decided to place a bid to sell power on the basis of the model, I would have sold at the peak.

**Next steps and improvements:**
- Training/testing the model on a larger country with more interconnects (eg. France, Germany)
- Inclusion of intraday pricing data and more live data (to counteract ENTSO-E's delayed publications). The absence of intraday pricing prevents direct modelling of the full market sequence and the model's ability to capture full short-term market assessments.
- Improved and live weather data at select nodes on a country basis. Clouds passing over solar installations are a major source of Imbalance volatility in Europe.
- Testing other ML techniques (eg. neural networks)
- Inclusion of live commodity data (eg. gas, coal, ETS)
- **Creating a battery trading algorithm based on the predicted price curves, charging and discharging into the DA and Imbalance markets. Then compare against revenue benchmarks.**
