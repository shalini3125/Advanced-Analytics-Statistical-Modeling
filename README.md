Task 4  - Advanced-Analytics-Statistical-Modeling

Timeline: 6 Days (Day 21–22 core focus: Statistical Analysis) Dataset: Sales data with columns — Order ID, Order Date, Customer Name, Region, City, Category, Sub-Category, Product Name, Quantity, Unit Price, Discount, Sales, Profit, Payment Mode

📋 Overview

This task covers 4 major parts:

Descriptive Statistics & Hypothesis Testing
Time Series Analysis
Clustering (Customer/Order Segmentation)
Predictive Modeling (Regression)

Each part is documented below with what was done, why, the code, and business implications.

Part 1: Descriptive Statistics & Hypothesis Testing
1.1 Descriptive Statistics

Goal: Summarize Sales and Profit using mean, median, mode, standard deviation, and skewness.

python
import pandas as pd
import numpy as np
from scipy import stats

df = pd.read_csv('your_filename.csv')

numeric_cols = ['Sales', 'Profit', 'Quantity', 'Unit Price', 'Discount']

summary = pd.DataFrame({
    'Mean': df[numeric_cols].mean(),
    'Median': df[numeric_cols].median(),
    'Mode': df[numeric_cols].mode().iloc[0],
    'Std Dev': df[numeric_cols].std(),
    'Skewness': df[numeric_cols].skew()
})
print(summary)

Result (Sales):

Metric	Value
Mean	₹106,733.20
Median	₹83,080
Std Dev	₹85,108
Skewness	0.953

Finding: Sales are right-skewed (skew = 0.953) — a small number of large orders pull the mean above the median. High std dev relative to mean shows wide variability in order size.

Business Implication: Don't rely on the mean alone for forecasting. Segment customers into "regular buyers" (near median) and "bulk/high-value buyers" (driving the mean up), and build separate strategies for each.

1.2 T-Test — Compare Two Groups (Sales: East vs West Region)

Goal: Test whether average Sales differs significantly between two regions.

python
group_a = df[df['Region'] == 'East']['Sales']
group_b = df[df['Region'] == 'West']['Sales']

t_stat, p_value = stats.ttest_ind(group_a, group_b, equal_var=False)
print(f"T-statistic: {t_stat:.3f}")
print(f"P-value: {p_value:.4f}")

Result: t = 0.747, p = 0.4552

Finding: Since p > 0.05, there is no statistically significant difference in Sales between East and West regions. Any observed difference is likely random variation.

Business Implication: Region does not significantly influence order value — no need for region-specific pricing/promotion strategy based on Sales amount alone. Investigate other factors (e.g., Category, Discount) instead.

1.3 Chi-Square Test — Categorical Relationship (Category vs Payment Mode)

Goal: Test whether product Category and Payment Mode are related (independent or not).

python
contingency_table = pd.crosstab(df['Category'], df['Payment Mode'])
chi2, p_value, dof, expected = stats.chi2_contingency(contingency_table)
print(f"Chi-square statistic: {chi2:.3f}")
print(f"P-value: {p_value:.4f}")

Result: chi2 = 41.894, p = 0.2304

Finding: Since p > 0.05, Category and Payment Mode are independent — no significant relationship. Payment mode choice doesn't depend on what category a customer buys.

Business Implication: No need for category-specific payment promotions (e.g., "Credit Card offers on Electronics only"). Checkout/payment experience can be standardized across all categories.

1.4 Confidence Interval (Sales)

Goal: Estimate a 95% confidence range for true average Sales.

python
data = df['Sales'].dropna()
mean = np.mean(data)
sem = stats.sem(data)
ci = stats.t.interval(0.95, len(data)-1, loc=mean, scale=sem)
print(f"Mean Sales: {mean:.2f}")
print(f"95% CI: ({ci[0]:.2f}, {ci[1]:.2f})")

Result: Mean = ₹106,733.20, 95% CI = (₹104,373.60, ₹109,092.81)

Finding: We are 95% confident the true average Sales value across all orders lies between ₹104,373.60 and ₹109,092.81. The narrow interval indicates a precise, reliable estimate.

Business Implication: Use ₹104,373.60 (lower bound) as a conservative benchmark for revenue forecasting and target-setting, to avoid over-promising based on the higher point estimate.

Part 2: Time Series Analysis
2.1 Convert to Time Series Format
python
df_ts = df.copy()
df_ts['Order Date'] = pd.to_datetime(df_ts['Order Date'])
df_ts = df_ts.set_index('Order Date')
df_ts = df_ts.sort_index()

Why: Converts the text date column into a real datetime index so date-based grouping and analysis become possible.

2.2 Resample Data (Daily / Weekly / Monthly)
python
daily_sales = df_ts['Sales'].resample('D').sum()
weekly_sales = df_ts['Sales'].resample('W').sum()
monthly_sales = df_ts['Sales'].resample('M').sum()

Why: Groups thousands of individual transactions into clean daily/weekly/monthly totals so trends are visible. Monthly is used for decomposition and forecasting (cleanest signal).

2.3 Decompose into Trend, Seasonality, Residuals
python
from statsmodels.tsa.seasonal import seasonal_decompose

result_monthly = seasonal_decompose(monthly_sales, model='additive', period=12)
result_monthly.plot()

Why: Splits the series into:

Trend — long-term direction (growth/decline)
Seasonality — repeating yearly patterns
Residual — random leftover noise

Note: Requires at least 24 months of data for period=12; use period=4 (quarterly) if less data is available.

2.4 Simple Moving Average Forecast
python
monthly_sales_ma = monthly_sales.rolling(window=3).mean()
forecast_next_month = monthly_sales[-3:].mean()
print(f"Forecast for next month: {forecast_next_month:.2f}")

Why: Averages the last 3 months to smooth out noise and produce a simple, defensible forecast baseline.

Business Implication: Use the moving average forecast for near-term revenue planning; adjust inventory/staffing ahead of seasonal peaks identified in decomposition.

Part 3: Clustering (Segmentation)
3.1 Prepare Features (Scale using StandardScaler)
python
from sklearn.preprocessing import StandardScaler

features = df[['Sales', 'Profit', 'Quantity', 'Discount']].dropna()
scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)

Why: Puts all columns on the same scale so no single column (e.g., Sales, which has large numbers) unfairly dominates the clustering.

3.2 Find Optimal K (Elbow Method)
python
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

inertia = []
for k in range(1, 11):
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    km.fit(scaled_features)
    inertia.append(km.inertia_)

plt.plot(range(1, 11), inertia, marker='o')
plt.xlabel('K')
plt.ylabel('Inertia')
plt.show()

Why: Tests different numbers of clusters (K) and helps visually identify the point ("elbow") where adding more clusters stops improving fit meaningfully.

3.3 Apply K-Means Clustering
python
optimal_k = 4  # chosen from elbow chart
kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
cluster_labels = kmeans.fit_predict(scaled_features)
features['Cluster'] = cluster_labels

Why: Assigns every row to one of K groups based on similarity across Sales, Profit, Quantity, and Discount.

3.4 Visualize Clusters using PCA (2D)
python
from sklearn.decomposition import PCA

pca = PCA(n_components=2)
pca_features = pca.fit_transform(scaled_features)

plt.scatter(pca_features[:, 0], pca_features[:, 1], c=cluster_labels, cmap='viridis')
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.show()

Why: Compresses 4 dimensions down to 2 so clusters can be visually inspected on a scatter plot.

3.5 Profile Each Segment
python
cluster_profile = features.groupby('Cluster')[['Sales', 'Profit', 'Quantity', 'Discount']].mean()
print(cluster_profile)

Why: Turns cluster numbers into business-readable personas (e.g., "Premium Buyers," "Discount-Driven Buyers," "Bulk Buyers," "Occasional Small Buyers") with tailored recommendations for each.

Part 4: Predictive Modeling (Regression)
4.1 Define Target Variable
python
target = 'Sales'
feature_cols = ['Quantity', 'Unit Price', 'Discount', 'Profit']
X = df[feature_cols]
y = df[target]

Why: Sales is the value being predicted (target); the remaining numeric columns are used as predictive features.

4.2 Train/Test Split (80/20)
python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

Why: Holds back 20% of data to fairly evaluate the model on data it has never seen, avoiding overfitting bias.

4.3 Build Linear Regression Model
python
from sklearn.linear_model import LinearRegression

model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

Why: Learns a formula relating Quantity, Unit Price, Discount, and Profit to Sales, then predicts Sales on unseen test data.

4.4 Evaluate Model (R², MAE, RMSE)
python
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
Metric	Meaning
R²	% of Sales variation explained by the model (closer to 1 = better)
MAE	Average prediction error in ₹
RMSE	Similar to MAE, penalizes large errors more heavily
4.5 Identify Top 3 Important Features
python
importance = pd.DataFrame({
    'Feature': feature_cols,
    'Coefficient': model.coef_
})
importance['Abs_Coefficient'] = importance['Coefficient'].abs()
importance = importance.sort_values(by='Abs_Coefficient', ascending=False)
print(importance.head(3))

Why: Ranks features by their influence (weight) on predicting Sales — tells the business which factors matter most.

✅ Task 4 Checklist
 Descriptive statistics (Sales, Profit)
 T-test (East vs West Sales)
 Chi-square test (Category vs Payment Mode)
 Confidence interval (Sales)
 Time series conversion, resampling, decomposition, moving average forecast
 Clustering: scaling, elbow method, K-Means, PCA visualization, segment profiling
 Regression: target/feature definition, train/test split, model build, evaluation, top features
📁 Suggested Project Structure
project/
├── data/
│   └── sales_data.csv
├── notebooks/
│   └── task4_analysis.ipynb
├── outputs/
│   ├── decomposition_plots.png
│   ├── elbow_method.png
│   ├── cluster_visualization.png
├── README.md
└── report.docx  (final documented findings)
Key Takeaways for Stakeholders
Sales are right-skewed — plan around the median, not just the mean.
Region and Payment Mode do not significantly affect Sales/Category choice — focus resources elsewhere.
95% CI gives a reliable revenue benchmark (₹104K–₹109K average order value).
Time series decomposition reveals trend and seasonal patterns useful for inventory/staffing planning.
Customer segments (clusters) enable targeted marketing instead of one-size-fits-all campaigns.
Regression model identifies which factors (Quantity, Unit Price, Discount, Profit) most strongly drive Sales — useful for pricing and promotion strategy
