# Business Performance Dashboard

## Project Overview

This project explores retail sales performance through an interactive dashboard built using Google Looker Studio. The goal was to organize transactional data into a clear visual format that helps users understand how sales and profit evolve over time and how performance differs across product categories and regions.

The dashboard is designed to help answer practical analytical questions such as how overall performance changes over time, which business segments contribute most to profitability, and whether higher sales consistently translate into stronger financial outcomes. By combining summary metrics with interactive visual analysis, the dashboard provides both a quick overview and the ability to explore performance in more detail.

---

## Data Source

The analysis uses the Sample Superstore dataset, a publicly available retail dataset obtained from Kaggle. The dataset contains order-level transaction records including sales values, profit, product categories, customer segments, order dates, and regional information.

Dataset source:  
[Sample Superstore Dataset – Kaggle](https://www.kaggle.com/datasets/vivek468/superstore-dataset-final)

The structure of the dataset makes it suitable for analyzing performance over time and comparing outcomes across business dimensions using financial metrics such as sales and profit.

---

## Steps & Methodology

The dataset was connected directly to Looker Studio, where fields were reviewed to ensure values were aggregated correctly. A set of core metrics was defined to summarize performance, including total sales, total profit, profit margin, total orders, and average order value.

The dashboard layout was designed to move from overview to detail. Key indicators appear first to provide context, followed by a trend chart showing performance over time. Category and regional charts help explain where results are coming from, while a scatter chart compares sales and profit at a more detailed level. A transaction table is included at the end to allow users to view the underlying records behind the visual summaries.

Filters were added so users can adjust the view by region, category, and date range, making the dashboard interactive rather than static.

---

## Dashboard Overview

<img src="images/dashboard overview.png" width="900">

---

## Key Insights

The analysis highlights several observable patterns. Technology products tend to generate stronger profitability compared to other categories, while performance varies across regions. The dashboard also shows that higher sales volumes do not always correspond to higher profit margins, suggesting differences in pricing or discount behavior across product groups.

---

## Live Dashboard Link

The interactive dashboard can be accessed [here](https://lookerstudio.google.com/s/iWrjxy-Ce8g)

## Assumptions & Limitations

The dataset represents example retail transactions rather than real business operations. Profit values reflect transaction-level results only and do not include additional operational costs. The analysis assumes the dataset is complete and consistent across all records.
