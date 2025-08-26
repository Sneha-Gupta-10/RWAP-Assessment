# RWAP Dashboard Analysis (2025–26) <br>
[**Dashboard Can Be Accessed Here**](https://rwap2025-26cx4-hrmm2xuikpwuunbmjzbycg.streamlit.app/) <br>


**Project Title:** Real-World Asset Valuation & Classification using GIS and Machine Learning  

**Team Members:**  
- 055001 – Aayush Garg  
- 055027 – Rushil Kohli  
- 055042 – Shagun Seth  
- 055047 – Sneha Gupta  

---

## Overview
This project develops a comprehensive **asset valuation and classification system** by integrating **U.S. Government Real Property Assets** with the **Zillow Housing Index**. Using **GIS, Machine Learning, and Spatial Analytics**, we created a pipeline that predicts valuations, clusters assets into market regimes, and visualizes insights through an interactive dashboard.

The study combines:
- Macro-level housing market data (Zillow)  
- Micro-level government asset data  
- Machine learning models  
- Geospatial analysis  
- Interactive dashboarding  

---

## Tools and Technologies
- **Programming:** Python (NumPy, Pandas, Scikit-learn, Matplotlib, Seaborn)  
- **GIS Libraries:** GeoPandas, Folium, Shapely, PySAL  
- **Visualization & Dashboard:** Streamlit, Folium (Web Maps), Plotly, Matplotlib  
- **Execution Environment:** Google Colab, Jupyter Notebook  

---

## Datasets

### 1. U.S. Government Real Property Assets
- Size: ~8,652 rows × 18 columns  
- Features: Asset IDs, Ownership, Location (City/State/ZIP, Latitude/Longitude), Rentable Sq Ft, Status, Type  
- Nature: Micro-level government asset inventory  

### 2. Zillow Housing Index
- Size: ~26,315 rows × 316 columns  
- Features: Region metadata (RegionID, RegionName, City, State, Metro, County) + Monthly Housing Prices (2000–2025)  
- Nature: Spatio-temporal housing valuation dataset  

---

## Objectives
- Build an **asset valuation engine** that estimates fair market value at the asset level.  
- Perform **geospatial enrichment** by linking government assets with Zillow indices.  
- Conduct **unsupervised segmentation** to identify natural asset clusters.  
- Develop **supervised prediction models** to forecast valuations for new or updated assets.  
- Deliver an **interactive dashboard** for asset exploration and decision-making.  

---

## Data Preprocessing

### Government Assets Dataset
- Standardized city/state names and ZIP codes (5-digit).  
- Converted latitude/longitude to numeric values.  
- Preserved construction year, asset type, and status.  

### Zillow Housing Index
- Cleaned and imputed missing values using **KNN Imputer**.  
- Engineered 16 features (mean/median prices, volatility, trend slope, recent averages).  
- Scaled features using **MinMaxScaler** with separate handling for `last_price`.  

---

## Modeling Approach

### Unsupervised Clustering
- Optimal number of clusters: **K=2** (Silhouette Score: 0.531).  
- Identified clusters:  
  - **High-Value Markets** – Coastal and metro regions with stronger appreciation.  
  - **Upper-Mid Markets** – Stable but slower growth regions.  

### Supervised Regression
- **Global Model:** Random Forest (R² ≈ 0.999, MAE ≈ 0.0002).  
- **Cluster Models:**  
  - Cluster 0 (High-Value): Random Forest (R² ≈ 0.986).  
  - Cluster 1 (Upper-Mid): Gradient Boosting (R² ≈ 0.9998).  

- **Feature Importance:**  
  - `last_price` (62%)  
  - `recent_6mo_avg` (19%)  
  - `recent_12mo_avg` (7%)  

---

## Key Insights
- **Primary valuation driver:** Recent housing prices, more than long-term historical trends.  
- **Spatial patterns:** Coastal states (CA, NY, MA) show higher valuations; Midwest and Southern states show lower valuations.  
- **Portfolio distribution:**  
  - Median predicted asset value: **$385,751**  
  - Mean predicted asset value: **$492,330**  
  - Maximum predicted asset value: **>$3.45 million**  
- **Spatial autocorrelation:** Moran’s I = **0.623 (p=0.001)**, indicating significant clustering of asset valuations.  

---

## Visualizations
- **Choropleth maps:** Median predicted asset values by state.  
- **Histograms:** Distribution of predicted valuations (right-skewed).  
- **Scatterplots:** Asset size vs. predicted valuations.  
- **PCA and t-SNE projections:** Clear visualization of cluster separations.  

---

## Results Summary

| Model        | Train R² | Validation R² | Test R² | MAE (scaled) |
|--------------|----------|---------------|---------|--------------|
| Global RF    | 0.9994   | 0.9991        | 0.9987  | 0.0002       |
| Cluster 0 RF | 0.9994   | 0.9982        | 0.9864  | 0.00095      |
| Cluster 1 GB | 1.0000   | 0.9728        | 0.9998  | 0.00046      |

---

## Conclusion
The project demonstrates that integrating macro housing indices with micro-level asset data produces highly accurate and interpretable asset valuations.  
- **Global Random Forest** provides robust baseline predictions.  
- **Cluster-specific models** enhance interpretability and domain relevance.  
- **Geospatial clustering** confirms economic disparities between coastal and inland markets.  

The results position this system as a scalable and defensible framework for **federal asset valuation, portfolio risk assessment, and strategic capital planning**.  
