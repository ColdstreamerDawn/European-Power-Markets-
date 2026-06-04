from entsoe import EntsoePandasClient
from entsoe.exceptions import NoMatchingDataError
import pandas as pd
from itertools import permutations
from xgboost import XGBRegressor
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import shap

#Initialisation of data request, uses personal api key
client = EntsoePandasClient(api_key="5de489ac-9680-449e-a85d-3db36070c128")
start = pd.Timestamp('20200603', tz='UTC') #start date of the downloaded set
end = pd.Timestamp('20260602', tz='UTC') #end date of the downloaded set

country_code = 'PT'
neighbours = ["ES","PT"]
other_tech = ["B02", "B04", "B05", "B06", "B10", "B11", "B12", "B14"]

# 0.0.0 Create function to take convert to quarter hourly UTC timestamp 
def make_quarter_hourly(df, ts_col="timestamp"):
  df = df.copy()
  df[ts_col] = pd.to_datetime(df[ts_col], utc=True)

  # Set timestamp as index
  df = df.set_index(ts_col).sort_index()

  # Keep only numeric columns
  numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
  df = df[numeric_cols]

  # Convert to 15-minute frequency
  # Hourly values are forward-filled into the four quarter-hours
  df = df.resample("15min").ffill()

  df.index.name = ts_col
  return df

# 0.0.1 Create function to add "timestamp" to first column and set as index, then adjust to hourly
def add_timestamp(df):
  df = df.reset_index()
  df = df.rename(columns={"index": "timestamp"})
  df = make_quarter_hourly(df)
  return df 

# 0.1.0 - RENEWABLES GENERATION FORECAST FUNCTION
dfs_RES =[]
def RES_forecast(country_code, startdate, enddate):

  #Make the request
  df = client.query_wind_and_solar_forecast(country_code, start=startdate, end=enddate, psr_type=None)
  df = add_timestamp(df)

  #Pick up any columns with "wind" in them and sum together for simplicity,then also add country code
  wind_cols = [c for c in df.columns if "wind" in c.lower()]
  df["Wind"] = df[wind_cols].sum(axis=1)
  df = df.drop(columns=wind_cols)
  df = df.rename(columns=lambda c: f"{country_code}_{c}")
  dfs_RES.append(df)

#0.1.1 - INTRADAY RENEWABLES GENERATION FORECAST FUNCTION (IMBALANCE ONLY)
dfs_RES_intraday = []
def RES_intraday_forecast(country_code, startdate, enddate):
  
  #Make the request
  try:
    df = client.query_intraday_wind_and_solar_forecast(country_code,start=startdate,end=enddate,
        psr_type=None)
    df = add_timestamp(df)

    #Pick up any columns with "wind" in them and sum together for simplicity,then also add country code
    wind_cols = [c for c in df.columns if "wind" in c.lower()]
    df["Wind"] = df[wind_cols].sum(axis=1)
    df = df.drop(columns=wind_cols)
    df = df.rename(columns=lambda c: f"{country_code}_intraday_{c}")
    dfs_RES_intraday.append(df)

  except NoMatchingDataError:
    print(f"NoMatchingDataError: No intraday RES forecast for {country_code}.")
    empty_df = pd.DataFrame()
    empty_df.index.name = "timestamp"
    dfs_RES_intraday.append(empty_df)

# 0.2.0 - POWER PRICE DATA FUNCTION
dfs_prices = []
def price_data(country_code, startdate, enddate):

  #Day-Ahead price request
  df_da = client.query_day_ahead_prices(country_code, start=startdate, end=enddate)
  df_da = add_timestamp(df_da)
  df_da = df_da.rename(columns={0: f"DA_price"})
  
  #Imbalance price request
  df_Imb = client.query_imbalance_prices(country_code, start=startdate, end=enddate)
  df_Imb = add_timestamp(df_Imb)
  df_Imb = df_Imb.rename(columns={"Long":  "Imb_long", "Short": "Imb_short"})
  
  #Concatenate DA and Imbalance prices
  df = pd.concat([df_da, df_Imb], axis=1)
  dfs_prices.append(df)

#0.2.1 - IMBALANCE VOLUMES FUNCTION
dfs_imbalance_volumes = []
def imbalance_volumes(country_code, startdate, enddate):

  #Imbalance volume request
  try:
    df = client.query_imbalance_volumes(country_code,start=startdate,end=enddate)
    df = add_timestamp(df)
    df = df.rename(columns=lambda c: f"{country_code}_imbalance_volume_{c}")
    dfs_imbalance_volumes.append(df)

  except NoMatchingDataError:
    print(f"NoMatchingDataError: No imbalance volumes for {country_code}.")
    empty_df = pd.DataFrame()
    empty_df.index.name = "timestamp"
    dfs_imbalance_volumes.append(empty_df)

# 0.3.0 - CROSS-BORDER EXCHANGESS DATA FUNCTION
dfs_exchanges = []
def cross_border_exchanges(country_code_from, country_code_to, startdate, enddate):
  try:
    #Make the request, and name the column based on flows
    df = client.query_scheduled_exchanges(country_code_from, country_code_to, start=startdate, end=enddate)
    df = add_timestamp(df)
    df = df.rename(columns={0: f"{country_code_from}_{country_code_to}"})
    dfs_exchanges.append(df)

  except NoMatchingDataError:
    print(f"NoMatchingDataError: No data for cross-border exchanges from {country_code_from} to {country_code_to} in the specified period.")
    # If no data, append an empty DataFrame to avoid errors in pd.concat later
    # The column name is important for later concatenation and feature selection
    empty_df = pd.DataFrame(columns=[f"{country_code_from}_{country_code_to}"])
    # Set the index name to 'timestamp' for consistency with other dataframes after add_timestamp
    empty_df.index.name = 'timestamp'
    dfs_exchanges.append(empty_df)

# 0.3.1 CROSS-BORDER Permutations function, iterates through "neighbours" permutations
def cross_border_permutations(neighbours, startdate, enddate):
 flows = list(permutations(neighbours, 2))
 for flow in flows:
  cross_border_exchanges(flow[0], flow[1], startdate, enddate)

# 0.4.0 LOAD FORECAST FUNCTION
dfs_load = []
def load_forecast(country_code, startdate, enddate):

  #Make the request
  df = client.query_load_forecast(country_code, start=startdate, end=enddate)
  df = add_timestamp(df)
  df = df.rename(columns={"Forecasted Load": f"{country_code}_load"})
  dfs_load.append(df)

# 1.0.0 Fetch data
for countries in neighbours:
  RES_forecast(countries, start, end)
  RES_intraday_forecast(countries, start, end)
  load_forecast(countries, start, end)
cross_border_permutations(neighbours, start, end) #iterates over neighbours
price_data(country_code, start, end) #Only takes target country prices
imbalance_volumes(country_code, start, end)

# 1.0.1 merge all dataframes in dfs along the timestamp column, and make timestamp index column
df = pd.concat(dfs_exchanges + dfs_prices + dfs_load + dfs_RES + dfs_RES_intraday 
               + dfs_imbalance_volumes, axis=1)
df = df.reset_index()
print(df.head(24))

#2.0.0 This function takes each market and trains/tests on it - NOTE TEST DURATION INPUT IS HERE
def run_price_model(df, target_col, target_name, test_date, test_days=6):
  
  df_model = df.copy()
  df_model["power_price"] = df_model[target_col]
  old_lags = [c for c in df_model.columns if "lag" in c.lower()]
  df_model = df_model.drop(columns=old_lags, errors="ignore")
  
  #2.0.1 Regular time-based variables
  df_model["hour"] = df_model["timestamp"].dt.hour
  df_model["day_of_week"] = df_model["timestamp"].dt.dayofweek
  df_model["month"] = df_model["timestamp"].dt.month
  df_model["weekend"] = (df_model["timestamp"].dt.weekday >= 5).astype(int)
  

  #2.0.2 Logic such that DA forecasts don't take less than 24h price notice
  #because DA prices are set the day before
  if target_name == "DA":
    df_model[f"{target_name}_lag_24h"] = df_model["power_price"].shift(96)
    df_model[f"{target_name}_lag_48h"] = df_model["power_price"].shift(192)
    df_model[f"{target_name}_lag_72h"] = df_model["power_price"].shift(288)
    df_model[f"{target_name}_lag_1w"] = df_model["power_price"].shift(672)
  else:
    df_model[f"{target_name}_lag_15m"] = df_model["power_price"].shift(1)
    df_model[f"{target_name}_lag_1h"] = df_model["power_price"].shift(4)
    df_model[f"{target_name}_lag_24h"] = df_model["power_price"].shift(96)
    df_model[f"{target_name}_lag_1w"] = df_model["power_price"].shift(672)

  # 2.0.3 Calculates residual load in the dataframe
  for countries in neighbours:
    df_model[f"{countries}_load_residual"] = (df_model[f"{countries}_load"]
        - df_model[f"{countries}_Wind"] - df_model[f"{countries}_Solar"])
  
  #2.0.4 Drop any columns that are entirely NaN and picked up as no matching data for cross border exchanges
  df_model = df_model.dropna(axis=1, how="all")

  #2.0.5 Calculates intraday RES forecast errors / revisions
  for countries in neighbours:
    if f"{countries}_intraday_Wind" in df_model.columns and f"{countries}_Wind" in df_model.columns:
      df_model[f"{countries}_Wind_revision"] = (df_model[f"{countries}_intraday_Wind"] - df_model[f"{countries}_Wind"])

    if f"{countries}_intraday_Solar" in df_model.columns and f"{countries}_Solar" in df_model.columns:
      df_model[f"{countries}_Solar_revision"] = (df_model[f"{countries}_intraday_Solar"] - df_model[f"{countries}_Solar"])
  
  #2.0.6 Adds lagged imbalance volume indicators
  imb_volume_cols = [c for c in df_model.columns if "imbalance_volume" in c.lower()]
  for col in imb_volume_cols:
    df_model[f"{col}_lag_15m"] = df_model[col].shift(1)
    df_model[f"{col}_lag_1h"] = df_model[col].shift(4)
    df_model[f"{col}_lag_24h"] = df_model[col].shift(96)
    df_model[f"{col}_lag_1w"] = df_model[col].shift(672)

  #2.0.7 Note that we are testing over a set number of days defined in the function
  test_end = test_date + pd.Timedelta(days=test_days)

  #2.0.8 The splitting of the training and testing datasets 
  train = df_model[df_model["timestamp"] < test_date].copy()
  test = df_model[(df_model["timestamp"] >= test_date)
      & (df_model["timestamp"] < test_end)].copy()

  imbalance_only_cols = [c for c in df_model.columns
      if ("intraday" in c.lower() or "revision" in c.lower() or "imbalance_volume" in c.lower())]
  if target_col == "DA_price":
    drop_cols = ["timestamp","power_price","DA_price","Imb_long","Imb_short"] + imbalance_only_cols
  else:
    drop_cols = ["timestamp","power_price",target_col,"Imb_long","Imb_short"]

  X_train = train.drop(columns=drop_cols, errors="ignore")
  X_test = test.drop(columns=drop_cols, errors="ignore")
  Y_train = train["power_price"]
  Y_test = test["power_price"]
  
  #2.1.0 The training of the XGBoost model 
  model = XGBRegressor(
      learning_rate=0.05,
      max_depth=5,
      min_child_weight=1,
      n_estimators=1000,
      n_jobs=-1,
      objective="reg:squarederror",
      tree_method="hist",
      random_state=42)
  model.fit(X_train, Y_train)
  
  #2.1.1 Testing the model
  Y_pred = model.predict(X_test)
  preds = pd.DataFrame({"timestamp": test["timestamp"],f"{target_name}_forecast": Y_pred})
  combined = pd.merge(preds, test[["timestamp", "power_price"]], on="timestamp").dropna(subset=["power_price"])
  
  #2.1.2 Model metrics
  rmse_value = np.sqrt(mean_squared_error(combined["power_price"],combined[f"{target_name}_forecast"]))
  mae = mean_absolute_error(combined["power_price"],combined[f"{target_name}_forecast"])
  r2 = r2_score(combined["power_price"],combined[f"{target_name}_forecast"])

  print(target_name)
  print("X_train shape:", X_train.shape)
  print("Y_train shape:", Y_train.shape)
  print(f"RMSE: {rmse_value:.2f}")
  print(f"MAE: {mae:.2f}")
  print(f"R2: {r2:.2f}")

  return model, preds, combined, X_test

#3.0.0 Input the test date
test_date = pd.Timestamp("2026-05-11", tz="UTC")

#3.1.1 The Day-Ahead price model
model_DA, preds_DA, combined_DA, X_test_DA = run_price_model(df, "DA_price", "DA", test_date)

#3.1.2 The Long Imbalance model
model_long, preds_long, combined_long, X_test_long = run_price_model(df, "Imb_long", "Long", test_date)

#3.1.3 The Short Imbalance model
model_short, preds_short, combined_short, X_test_short = run_price_model(df, "Imb_short", "Short", test_date)

#4.0.0 The Charts
plot_sets = [
    ("Day-Ahead Price", combined_DA, "DA_forecast", model_DA, X_test_DA),
    ("Long Imbalance Price", combined_long, "Long_forecast", model_long, X_test_long),
    ("Short Imbalance Price", combined_short, "Short_forecast", model_short, X_test_short),]

for title, data, forecast_col, model, X_test in plot_sets:

    plt.figure(figsize=(15, 5))
    plt.plot(data["timestamp"], data["power_price"], label="Actual")
    plt.plot(data["timestamp"], data[forecast_col], label="Forecast")
    plt.title(title)
    plt.xlabel("Timestamp")
    plt.ylabel("€/MWh")
    plt.legend()
    plt.show()

    explainer = shap.TreeExplainer(model)
    shap_values = explainer(X_test)

    shap.summary_plot(shap_values, X_test, plot_type="bar")
