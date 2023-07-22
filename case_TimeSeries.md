# **Investigating the media attention impact on dispensing contraceptives in Australia**

## **[Keyword]**

<ins>Public Health</ins>, <ins>Interrupted Time Series</ins>, <ins>ARIMA</ins>, <ins>Exploratory Data Analysis</ins>, <ins>Data Visualization</ins>, <ins>R Programming</ins>

## **[Overview]**
I **analyzed the media attention impact on dispensing contraceptives** (combined/simple contraceptives) using **R**. The dataset consisted of monthly rates (per 1000 women of reproductive age) of PBS-subsidized dispensing of combined and simple contraceptives between January 2013 and December 2016. The media attention peaked in the last week of May 2015.

## **[Approach]**

- **Exploratory data analysis** (EDA) by decomposing each time series data (i.e., the combined or simple) to observe the trend, seasonality, outliers, stationarity, and autocorrelation
- **Log-transformation** of the data for eliminating autocorrelation and non-stationarity
- **Model selection** for each data based on the EDA (e.g., stationarity + no-autocorrelation à segmented time series, no-stationarity + autocorrelation à ARIMA)
- **Model fitting** for each time series by iteratively testing different parameters
- **Evaluation of time series changes**: step (interruption) and slope after media attention (= intervention).
- **Quantifying the above changes in tables** and **visualized by the actual time series against the counterfactual** (simulative plot if no intervention was present)

## **[Outcome]**
The media impact was agreeable on the combined contraceptives based on the changes (in %) with the monthly dispensing rate confidence intervals from the step change. I **achieved distinction** for this task.