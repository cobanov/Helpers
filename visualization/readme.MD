
## Outliers

```python
def get_outliers(col_name):
    
    clf = LocalOutlierFactor(n_neighbors=15)
    preds = clf.fit_predict(np.array(df_processed[col_name]).reshape(-1,1))

    preds_class = ["ok" if i == 1 else "outlier" for i in preds]
    df_processed["outlier"] = preds_class
    #df_processed.to_parquet("data_outlier.parquet")

def ee_outliers(col_name):
    
    ee = EllipticEnvelope()
    ee_preds = ee.fit_predict(np.array(df_processed[col_name]).reshape(-1,1))

    ee_preds_class = ["ok" if i == 1 else "ee_outlier" for i in ee_preds]
    df_processed["ee_outlier"] = ee_preds_class
    #df_processed.to_parquet("data_outlier.parquet")

```

## Visualization

### Distribution
```python
def dact_dist(dataset, high_corrs, class_col):
    
    """
    :dataset: pandas dataframe
    :values: columns to visualize
    :class_col: classes
    """
    
    labels = dataset[class_col].value_counts().index.to_list()
    for col_name in high_corrs:
        fig, ax = plt.subplots(figsize=(30,10))
        for label in labels: 
            sns.distplot(dataset[col_name][dataset[class_col]==label], ax=ax)
            ax.legend(labels)
        plt.show()
```

### Scatter

```python
def dact_scatter(dataset, target:str, cols_vis:list, class_col, std_thresh=2.5):
    
    
    for col_name in cols_vis:
        
        if class_col == "outlier":
            get_outliers(col_name)
            
        if class_col == "ee_outlier":
            ee_outliers(col_name)
        
        
        #RED LINES
        s3 = (dataset[col_name].mean()) + (std_thresh * dataset[col_name].std())
        s3m = (dataset[col_name].mean()) - (std_thresh * dataset[col_name].std())
        
        #QUANTILE
        q1=dataset[col_name].quantile(.25)
        q3 = df_processed[col_name].quantile(.75)

        iqr =  q3 - q1

        lowlim = q1 - 1.5 * iqr
        uplim = q3 + 1.5 * iqr


        fig, ax = plt.subplots(figsize=(30,10))
        
        ax.axhline(s3, color="red", linestyle="--")
        ax.axhline(s3m, color="red", linestyle="--")
        
        ax.axhline(lowlim, color="blue", linestyle="-", alpha=0.5)
        ax.axhline(uplim, color="blue", linestyle="-", alpha=0.5)
        
        labels = dataset[class_col].value_counts().index.to_list()
        
        #PLOT
        sns.scatterplot(data=dataset, y=col_name, x=target, hue=class_col)
        plt.show()