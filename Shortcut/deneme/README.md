
# Coding Topics To Be Implemented
* Data exploration (Cagatay)
    * Relationship with numerical variables - scatter plot
    * Relationship with categorical features - box plot
    * Scatter matrix
    * Correlation matrix
    * Histogram (distplot))

* Data Preprocessing (Yigitcan, Muratcan)
    *	Data cleansing
    *	Missing value
    *	Remove outlier
    *	Normalize data
    *	Convert categorical to dummy

* Model Creation (Mert, Ezgi, Muhammet)
    * Regression(XGBReg, LGBReg, Linear Regres)   (Ezgi)
    * Classification(RDF, XGBoost, DNN(Gpu optional)) (Muhammet)
    * Cross validation
    * Data separation
    * Hyper parameter tuning

* Analysis / Evaluation
    * classification (Aziz)
        * Confusion matrix
        * Accuracy
        * F score
    * Regression (Ezgi)
        * Rmse
        * R Squared (R²)
        * Shap Analysis (Yigitcan)
    * Bias/Variance (Ezgi)

# Define Function
    <!--drop useless columns such as ErrorBit-->
    drop useless columns such as ErrorBit
    df = df[df.columns.drop(list(df.filter(regex="Unnamed")))]
    df = df[df.columns.drop(list(df.filter(regex="SeriesLine")))]
    df = df[df.columns.drop(list(df.filter(regex='TypeNumber')))]
    df = df[df.columns.drop(list(df.filter(regex='ErrorBit')))]
    df = df[df.columns.drop(list(df.filter(regex='Dmc')))]
    '''process cilere sorulacaklar'''
    df = df[df.columns.drop(list(df.filter(regex='SpcResultStruct')))] 


    def dropColsStartingWithText(df, text_list):
    '''
    dropColsStartingWithText drop cols starting with text in text_list
    df : dataframe to drop columns
    text_list: potential textlist including texts to look for on df
    '''

        for text in text_list:
            df = df[df.columns.drop(list(df.filter(regex=text)))]

        return df



    if __name__ == "__main__":
        text_list = ["Unnamed","SeriesLine", "TypeNumber"]
        df= pd.Dataframe()
        dropColsStartingWithText(df, text_list)

# Unit test Script
All functions also have test fucntions which are named corresponds to function name \
* for example:\
    def test_dropColsStartingWithText():\
            > text_list = ["Unnamed","SeriesLine", "TypeNumber"]\
            > df= pd.Dataframe()\
            > dropColsStartingWithText(df, text_list)

# Pushing Concept
Before Pushing the codes gitlab please check that
 * all unit tests are written
 * all unit tests are succesfull

