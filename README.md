# Business Performance Dashboard

## Project Overview

This project explores business performance using an interactive dashboard built on retail sales transaction data. The goal was to organize raw operational data into a format that makes performance easier to understand and explore through visual analysis.

The dashboard focuses on providing a clear overview of revenue and profitability while allowing users to examine how results differ across product categories and regions. Instead of presenting isolated charts, the dashboard was designed to guide users from a high-level summary toward more detailed analysis, helping them understand both overall outcomes and the factors influencing performance.

---

## Data Source

The analysis uses the **Sample Superstore dataset**, a publicly available dataset commonly used for learning and practicing business intelligence workflows. The dataset represents simulated retail transactions and includes information such as order dates, sales values, profit, product categories, customer segments, and regional markets.

Dataset source:  
[Sample Superstore Dataset – Kaggle](https://www.kaggle.com/datasets/vivek468/superstore-dataset-final)

This structure makes it suitable for exploring sales performance, profitability patterns, and operational comparisons across different business dimensions.

---

## Steps & Methodology

The dataset was connected directly to Google Looker Studio, where fields were reviewed to confirm correct data types and aggregation behavior. Key performance indicators were then defined to summarize overall activity, including total sales, total profit, profit margin, total orders, and average order value.

The dashboard layout follows a simple analytical flow. Summary metrics appear first to provide context, followed by a time-based visualization showing how performance changes over time. Additional charts break down results by category and region to help explain differences in performance. A scatter chart was added to compare sales and profit at a more detailed level, making it easier to observe variations in profitability between product segments. A transaction table is included to allow users to view the underlying records behind the visual summaries.

Filters were implemented to allow users to interact with the dashboard by selecting regions, categories, or time periods, making the analysis flexible and exploratory.

---

## Dashboard Overview

<img src="images/dashboard overview.png" width="900">

---

## Key Insights

Several patterns can be observed from the dashboard. Technology products generally show stronger profitability compared to other categories, while performance varies between regions. The analysis also shows that higher sales do not always lead to higher profit, suggesting that discounting or cost differences may influence margins across products.

These observations demonstrate how interactive dashboards can help surface patterns that may not be immediately visible in raw data.

---

## Live Dashboard Link

The interactive dashboard can be accessed [Sample Superstore Dataset – Kaggle](https://lookerstudio.google.com/s/iWrjxy-Ce8g).

## Assumptions & Limitations

The dataset represents simulated retail operations and therefore does not include external business factors such as operational costs, logistics expenses, or market conditions. Profit values reflect transactional profitability rather than complete financial accounting measures. Additionally, the analysis assumes consistent data quality across regions and time periods.
