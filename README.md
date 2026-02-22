# Business Performance Dashboard

## Project Overview

This project presents an interactive business performance dashboard developed to analyze retail sales data and evaluate overall operational performance. The goal of the dashboard is to organize transactional data into clear visual insights that help users understand revenue trends, profitability, and performance differences across products and regions.

The dashboard provides a structured overview of business activity through key performance indicators and supporting visualizations. By combining summarized metrics with interactive charts, the solution allows users to explore performance patterns and better understand how different business segments contribute to overall results.


---

### Dashboard Overview

<img src="images/dashboard overview.png" width="900">

---

## Data Source

The analysis is based on the **Sample Superstore dataset**, a publicly available dataset commonly used for business intelligence and analytics practice. The dataset represents simulated retail transactions and includes information about orders, sales values, profit, product categories, customer segments, and geographic regions.

Dataset source:  
[Sample Superstore Dataset – Kaggle](https://www.kaggle.com/datasets/vivek468/superstore-dataset-final)

The dataset structure enables performance analysis over time as well as comparisons between categories and regions using financial metrics such as sales and profit.

---

The project began with connecting and reviewing the dataset in Google Looker Studio to confirm correct data types and aggregation behavior. Key performance indicators were defined to summarize business outcomes, including total sales, total profit, profit margin, total orders, and average order value.

A clear visual hierarchy guided the dashboard design. Summary indicators are presented at the top to provide an immediate overview, followed by a time-series visualization showing performance trends over time. Additional charts analyze results by product category and region to help explain differences in performance. A scatter visualization was included to explore the relationship between sales and profitability at the sub-category level. A detailed transaction table allows users to view the underlying records supporting the analysis.

Interactive filters were added to enable exploration of results by region, category, and date range, allowing users to adjust the analysis according to specific interests or questions.

## Key Insights

The dashboard highlights several observable patterns within the dataset. Technology products tend to show stronger profitability compared to other categories, while regional performance varies across markets. The analysis also shows that higher sales volumes do not always correspond to higher profit margins, suggesting differences in cost structure or discounting behavior across products.

These observations demonstrate how visual analytics can help identify performance patterns and support further business investigation.

---

## Live Dashboard Link

The interactive dashboard can be accessed through the following link:

**Live Dashboard:**  
[(https://lookerstudio.google.com/s/iWrjxy-Ce8g)]

## Assumptions & Limitations

The dataset represents simulated retail operations and therefore does not include external business factors such as operational costs, logistics expenses, or market conditions. Profit values reflect transactional profitability rather than complete financial accounting measures. Additionally, the analysis assumes consistent data quality across regions and time periods.
