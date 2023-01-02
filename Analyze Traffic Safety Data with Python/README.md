# Analyze Traffic Safety Data with Python

### Try some of these resources for extra help as you work:

* [View the Analyze Traffic Safety Data with Python cheatsheet](https://www.codecademy.com/learn/case-study-analyze-traffic-safety/modules/traffic-safety-case-study/cheatsheet)
* [View the solution notebook](./solution.html)
* [Learn more about analyzing traffic safety data in this introductory article](https://www.codecademy.com/courses/case-study-analyze-traffic-safety/articles/analyze-traffic-safety-data-with-python-article)


```python
import pandas as pd
import datetime as dt
from scipy.stats import pearsonr
from sklearn.linear_model import LinearRegression
import seaborn as sns
import matplotlib.pyplot as plt


%matplotlib inline
# set plot theme and palette
sns.set_theme()
sns.set_palette('colorblind')
```

## Traffic data exploration

### 1. Inspect the traffic safety dataset

After running the first cell to load all necessary libraries, we need to load our dataset. Using pandas, load the dataset `traffic.csv` and save it as `traffic`. Inspect the first few rows.


```python
# load dataset
traffic = pd.read_csv('traffic.csv')

# inspect first few rows
traffic.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Date</th>
      <th>Crashes_per_100k</th>
      <th>Season</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2006-01-01</td>
      <td>169.176541</td>
      <td>Winter</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2006-02-01</td>
      <td>154.028836</td>
      <td>Winter</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2006-03-01</td>
      <td>159.930002</td>
      <td>Spring</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2006-04-01</td>
      <td>155.741270</td>
      <td>Spring</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2006-05-01</td>
      <td>168.179208</td>
      <td>Spring</td>
    </tr>
  </tbody>
</table>
</div>



### 2. Inspect and format data types

The `traffic` data frame contains three columns: `Date`, `Crashes_per_100k`, and `Season`. In order to plot the `Crashes_per_100k` column as a time series, we need to make sure that the `Date` column is in date format. Inspect the data types in the data frame, convert the `Date` column to date format, and inspect the data types a second time.


```python
# inspect data types
traffic.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 180 entries, 0 to 179
    Data columns (total 3 columns):
     #   Column            Non-Null Count  Dtype  
    ---  ------            --------------  -----  
     0   Date              180 non-null    object 
     1   Crashes_per_100k  180 non-null    float64
     2   Season            180 non-null    object 
    dtypes: float64(1), object(2)
    memory usage: 4.3+ KB


<details>
    <summary style="display:list-item; font-size:16px; color:blue;"><i>What did we discover in this step? Toggle to check!</i></summary>

The `traffic` dataframe consists of 3 columns and 180 rows. Luckily, there are no missing data to contend with. The number of crashes is normalized to the annual population per 100,000 people. We will also need to format the `Date` variable since Python does not yet recognize it as a datetime variable.


</details>

Convert the `Date` column to the date datatype using the `pd.to_datatime(column)` function.


```python
# convert Date to date format
traffic['Date'] = pd.to_datetime(traffic['Date'])

# inspect data types
traffic.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 180 entries, 0 to 179
    Data columns (total 3 columns):
     #   Column            Non-Null Count  Dtype         
    ---  ------            --------------  -----         
     0   Date              180 non-null    datetime64[ns]
     1   Crashes_per_100k  180 non-null    float64       
     2   Season            180 non-null    object        
    dtypes: datetime64[ns](1), float64(1), object(1)
    memory usage: 4.3+ KB


<details>
    <summary style="display:list-item; font-size:16px; color:blue;"><i>What did we discover in this step? Toggle to check!</i></summary>

By using the `pd.to_datetime()` function, we converted a character string object to the `datetime64[ns]` datatype. This will allow us to plot a time series of data points.

</details>

### 3. Visualize traffic safety data
To get a sense of trends that may exist in the data, use seaborn's `sns.lineplot()` function to create a line plot of the `traffic` data with `Date` on the x-axis and `Crashes_per_100k` on the y-axis.


```python
# create line plot
sns.lineplot(x='Date', y='Crashes_per_100k', data=traffic)
```




    <AxesSubplot:xlabel='Date', ylabel='Crashes_per_100k'>




    
![png](README_files/README_12_1.png)
    


<details>
    <summary style="display:list-item; font-size:16px; color:blue;"><i>What did we discover in this step? Toggle to check!</i></summary>

Looking at the line plot of our collision data, we can see the decreasing trend in crash rates from 2006 continuing until 2010 or 2011, and then crash rates begin increasing. The data for 2020 is very different from the preceding years.
    
There also appear to be cyclical patterns, which may indicate differing crash rates by season.

</details>

### 4. Visualize seasonal rates

Since we saw a fair amount of variance in the number of collisions occurring throughout the year, we might hypothesize that the number of collisions increases or decreases during different seasons. We can visually explore this with a box plot. 

Use `sns.boxplot()` with crash rate on the x-axis and season on the y-axis. Remove the anomolous 2020 data by adjusting the `data` parameter to `traffic[traffic.Date.dt.year != 2020]`.


```python
# create box plot by season
sns.boxplot(x='Crashes_per_100k', y='Season',data=traffic[traffic.Date.dt.year != 2020])
```




    <AxesSubplot:xlabel='Crashes_per_100k', ylabel='Season'>




    
![png](README_files/README_15_1.png)
    


<details>
    <summary style="display:list-item; font-size:16px; color:blue;"><i>What did we discover in this step? Toggle to check!</i></summary>

Winter and Fall appear to have generally higher crash rates than Spring and Summer. Seasons may be the reason for the pattern in crash rates.

</details>

## Smartphone data exploration

### 5. Inspect the smartphone use dataset

The dataset `crashes_smartphones.csv` contains smartphone data from Pew Research Center matched to normalized crash rates from the `traffic` data frame for the years 2011 to 2019.

<details>
    <summary style="display:list-item;"><b>Toggle for an overview of the variables in this dataset.</b></summary>

* `Month_Year`: a shortened date with only the month and year of the survey
* `Crashes_per_100k`: the normalized crash rate matching the month and year of the smartphone usage survey
* `Season`: Winter, Spring, Summer, or Fall
* `Smartphone_Survey_Date`: the actual date the smartphone usage survey was conducted
* `Smartphone_usage`: the percent of survey participants that owned and used a smartphone

</details>

Load the dataset as `smartphones` and inspect the first few rows.


```python
# import dataset
smartphones = pd.read_csv('crashes_smartphones.csv')

# inspect first few rows
smartphones.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Month_Year</th>
      <th>Crashes_per_100k</th>
      <th>Season</th>
      <th>Smartphone_Survey_Date</th>
      <th>Smartphone_usage</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Apr-12</td>
      <td>133.213685</td>
      <td>Spring</td>
      <td>4/3/12</td>
      <td>46</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Apr-15</td>
      <td>150.077792</td>
      <td>Spring</td>
      <td>4/12/15</td>
      <td>67</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Apr-16</td>
      <td>172.401948</td>
      <td>Spring</td>
      <td>4/4/16</td>
      <td>72</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Aug-12</td>
      <td>145.403147</td>
      <td>Summer</td>
      <td>8/5/12</td>
      <td>44</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Dec-12</td>
      <td>169.160811</td>
      <td>Winter</td>
      <td>12/9/12</td>
      <td>45</td>
    </tr>
  </tbody>
</table>
</div>



### 6. Format date data type
Similar to the  `traffic` data frame, the `smartphones` data frame has a date column that is not properly formatted. Convert the `Smartphone_Survey_Date` column to the date data type using the `pd.to_datetime()` function and then inspect the data types in the data frame.


```python
# change to datetime object
smartphones['Smartphone_Survey_Date'] = pd.to_datetime(smartphones['Smartphone_Survey_Date'])

# inspect data types
smartphones.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 28 entries, 0 to 27
    Data columns (total 5 columns):
     #   Column                  Non-Null Count  Dtype         
    ---  ------                  --------------  -----         
     0   Month_Year              28 non-null     object        
     1   Crashes_per_100k        28 non-null     float64       
     2   Season                  28 non-null     object        
     3   Smartphone_Survey_Date  28 non-null     datetime64[ns]
     4   Smartphone_usage        28 non-null     int64         
    dtypes: datetime64[ns](1), float64(1), int64(1), object(2)
    memory usage: 1.2+ KB


<details>
    <summary style="display:list-item; font-size:16px; color:blue;"><i>What did we discover in this step? Toggle to check!</i></summary>

There is a lot less data available for smartphone usage rates than there was for crash rates. The `smartphones` dataframe consists of 5 columns and 28 rows. Luckily, there are no missing data to contend with.


</details>

### 7. Visualize smartphone use data
Now let's take a look at smartphone use over time. Create a line plot of the `smartphones` data with `Smartphone_Survey_Date` on the x-axis and `Smartphone_usage` on the y-axis.


```python
# create line plot
sns.lineplot(x='Smartphone_Survey_Date', y='Smartphone_usage', data=smartphones)
plt.show()
```


    
![png](README_files/README_25_0.png)
    


<details>
    <summary style="display:list-item; font-size:16px; color:blue;"><i>What did we discover in this step? Toggle to check!</i></summary>

We can see a trend of smartphone usage increasing over time.

</details>

## Relationship exploration

### 8. Visualize crash rate by smartphone use
A scatter plot with smartphone usage on one axis and crash rates on the other axis will give us an idea of whether there is a relationship between these two variables. 

Create a scatter plot with a regression line using seaborn's `sns.regplot()` with `Smartphone_usage` on the x-axis and `Crashes_per_100k` on the y-axis.


```python
# create scatter plot with regression line
sns.regplot(x='Smartphone_usage', y='Crashes_per_100k', data=smartphones)
plt.show()
```


    
![png](README_files/README_29_0.png)
    


<details>
    <summary style="display:list-item; font-size:16px; color:blue;"><i>What did we discover in this step? Toggle to check!</i></summary>

There appears to be a positive linear relationship between the rate of car crashes and the rate of adult smartphone usage in the U.S.

</details>

### 9. Check the correlation coefficient

To test whether the correlation between `Smartphone_usage` and `Crashes_per_100k` is statistically significant, we can calculate the Pearson's _r_ correlation coefficient and the associated _p_-value. 

Use `corr, p = pearsonr(column1, column2)` on the `Smartphone_usage` and `Crashes_per_100k` columns in the `smartphones` dataframe. Then use the provided code to print `corr` and `p` to see the results.


```python
# find Pearson's r and p-value
corr, p = pearsonr(smartphones['Smartphone_usage'], smartphones['Crashes_per_100k'])

# print corr and p
print("Pearson's r =",  round(corr,3))
print("p = ", round(p,3))
```

    Pearson's r = 0.513
    p =  0.005


<details>
    <summary style="display:list-item; font-size:16px; color:blue;"><i>What did we discover in this step? Toggle to check!</i></summary>

The Pearson's r correlation coefficient is greater than 0.5, which indicates a moderately strong positive relationship. The p-value is less than 0.05. Together, this tells us that there is a statistically significant correlation between adult smartphone usage rates and car crash rates in the U.S. We have to be careful though: correlation does not mean causation, as the saying goes. Many other factors may be contributing to the rise in car crash rates from 2011 to 2019. 

</details>

## Analysis

### 10. Run a linear regression
We can use a linear regression to predict crash rates based on smart phone usage. Let's regress crash rates on smartphone usage. Then we can predict the crash rate in 2020 and see if it matches the actual crash rate in 2020!

We have provided the code to convert the variables to NumPy arrays that will work with the modeling function. The `Smartphone_usage` array is saved as `X`, and the `Crashes_per_100k` array is saved as `y`.

Initiate the model by saving `LinearRegression()` to the variable `lm`. Then fit the model and run the regression with `.fit()`.


```python
# convert columns to arrays
X = smartphones['Smartphone_usage'].to_numpy().reshape(-1, 1)
y = smartphones['Crashes_per_100k'].to_numpy().reshape(-1, 1)
```


```python
# initiate the linear regression model
lm = LinearRegression()

# fit the model
lm.fit(X, y)
```




<style>#sk-container-id-1 {color: black;background-color: white;}#sk-container-id-1 pre{padding: 0;}#sk-container-id-1 div.sk-toggleable {background-color: white;}#sk-container-id-1 label.sk-toggleable__label {cursor: pointer;display: block;width: 100%;margin-bottom: 0;padding: 0.3em;box-sizing: border-box;text-align: center;}#sk-container-id-1 label.sk-toggleable__label-arrow:before {content: "▸";float: left;margin-right: 0.25em;color: #696969;}#sk-container-id-1 label.sk-toggleable__label-arrow:hover:before {color: black;}#sk-container-id-1 div.sk-estimator:hover label.sk-toggleable__label-arrow:before {color: black;}#sk-container-id-1 div.sk-toggleable__content {max-height: 0;max-width: 0;overflow: hidden;text-align: left;background-color: #f0f8ff;}#sk-container-id-1 div.sk-toggleable__content pre {margin: 0.2em;color: black;border-radius: 0.25em;background-color: #f0f8ff;}#sk-container-id-1 input.sk-toggleable__control:checked~div.sk-toggleable__content {max-height: 200px;max-width: 100%;overflow: auto;}#sk-container-id-1 input.sk-toggleable__control:checked~label.sk-toggleable__label-arrow:before {content: "▾";}#sk-container-id-1 div.sk-estimator input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-1 div.sk-label input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-1 input.sk-hidden--visually {border: 0;clip: rect(1px 1px 1px 1px);clip: rect(1px, 1px, 1px, 1px);height: 1px;margin: -1px;overflow: hidden;padding: 0;position: absolute;width: 1px;}#sk-container-id-1 div.sk-estimator {font-family: monospace;background-color: #f0f8ff;border: 1px dotted black;border-radius: 0.25em;box-sizing: border-box;margin-bottom: 0.5em;}#sk-container-id-1 div.sk-estimator:hover {background-color: #d4ebff;}#sk-container-id-1 div.sk-parallel-item::after {content: "";width: 100%;border-bottom: 1px solid gray;flex-grow: 1;}#sk-container-id-1 div.sk-label:hover label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-1 div.sk-serial::before {content: "";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: 0;}#sk-container-id-1 div.sk-serial {display: flex;flex-direction: column;align-items: center;background-color: white;padding-right: 0.2em;padding-left: 0.2em;position: relative;}#sk-container-id-1 div.sk-item {position: relative;z-index: 1;}#sk-container-id-1 div.sk-parallel {display: flex;align-items: stretch;justify-content: center;background-color: white;position: relative;}#sk-container-id-1 div.sk-item::before, #sk-container-id-1 div.sk-parallel-item::before {content: "";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: -1;}#sk-container-id-1 div.sk-parallel-item {display: flex;flex-direction: column;z-index: 1;position: relative;background-color: white;}#sk-container-id-1 div.sk-parallel-item:first-child::after {align-self: flex-end;width: 50%;}#sk-container-id-1 div.sk-parallel-item:last-child::after {align-self: flex-start;width: 50%;}#sk-container-id-1 div.sk-parallel-item:only-child::after {width: 0;}#sk-container-id-1 div.sk-dashed-wrapped {border: 1px dashed gray;margin: 0 0.4em 0.5em 0.4em;box-sizing: border-box;padding-bottom: 0.4em;background-color: white;}#sk-container-id-1 div.sk-label label {font-family: monospace;font-weight: bold;display: inline-block;line-height: 1.2em;}#sk-container-id-1 div.sk-label-container {text-align: center;}#sk-container-id-1 div.sk-container {/* jupyter's `normalize.less` sets `[hidden] { display: none; }` but bootstrap.min.css set `[hidden] { display: none !important; }` so we also need the `!important` here to be able to override the default hidden behavior on the sphinx rendered scikit-learn.org. See: https://github.com/scikit-learn/scikit-learn/issues/21755 */display: inline-block !important;position: relative;}#sk-container-id-1 div.sk-text-repr-fallback {display: none;}</style><div id="sk-container-id-1" class="sk-top-container"><div class="sk-text-repr-fallback"><pre>LinearRegression()</pre><b>In a Jupyter environment, please rerun this cell to show the HTML representation or trust the notebook. <br />On GitHub, the HTML representation is unable to render, please try loading this page with nbviewer.org.</b></div><div class="sk-container" hidden><div class="sk-item"><div class="sk-estimator sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-1" type="checkbox" checked><label for="sk-estimator-id-1" class="sk-toggleable__label sk-toggleable__label-arrow">LinearRegression</label><div class="sk-toggleable__content"><pre>LinearRegression()</pre></div></div></div></div></div>



### 11. Print and interpret regression coefficients

Let's see the values our model produced. Print the coefficients from our `lm` model. Then think about which parts of the regression line equation these values represent.


```python
# print the coefficients 
print("Coef: \n", lm.intercept_, lm.coef_)
```

    Coef: 
     [120.6637106] [[0.66103316]]


<details>
    <summary style="display:list-item; font-size:16px; color:blue;"><i>What did we discover in this step? Toggle to check!</i></summary>

The generic equation for a line is `y = b + (m * x)`, where `b` is the value where the the line intercepts the y-axis and `m` is the slope of the line. In this step, we learned the two coefficients of our linear model, `b = 120.6637` and `m = 0.6610`. So the equation for our linear model is `y = 120.6637 + (0.6610 * x)` and we can use this equation to predict new values of y from any new value of x.
    
We can also interpret the slope of 0.6610: every additional percentage point of smartphone usage is associated with an additional 0.6610 crashes per 100,000 people.
</details>

### 12. Make a prediction

Let's assume smartphone usage was the same for 2020 as it was for 2019. This is a reasonable asssumption since the increase in smartphone usage that we observed in our plot started to plateau at the end of the time series. Let's use this approximation and our regression model to predict the crash rate in 2020.

From our model output, the regression line equation is `Crashes_per_100k = 120.6637 + (0.6610 * Smartphone_usage)`. Run the provided code to view the smartphone usage rate for 2019. Then substitute this value into the equation, using Python as a calculator to predict the crash rate for 2020.


```python
# get the smartphone usage rate from 2019
smartphones[smartphones['Month_Year'] == "Feb-19"].Smartphone_usage
```




    7    81
    Name: Smartphone_usage, dtype: int64




```python
# predict the crash rate in 2020 using the regression equation
Crashes_per_100k = 120.6637 + (0.6610 * 81)
print(Crashes_per_100k)
```

    174.2047


<details>
    <summary style="display:list-item; font-size:16px; color:blue;"><i>What did we discover in this step? Toggle to check!</i></summary>

If the smartphone usage rate was the same in 2020 as in 2019 (81%), our model predicts that the crash rate in 2020 would be 174.205 crashes per 100,000 people.  

</details>

### 13. Compare to the actual rate

How good was our prediction? Get the actual crash rate for February of 2020 from the `traffic` dataframe using `pd.to_datetime("2020-02-01")` as the value for `Date`.


```python
# get the actual crash rate in Feb 2020
traffic[traffic['Date'] == pd.to_datetime("2020-02-01")].Crashes_per_100k
```




    169    157.88955
    Name: Crashes_per_100k, dtype: float64



<details>
    <summary style="display:list-item; font-size:16px; color:blue;"><i>What did we discover in this step? Toggle to check!</i></summary>

The actual crash rate in Februrary of 2020 was only 157.8895, which was a fair bit lower than our prediction. 

</details>

### 14. Visualize the prediction

Let's plot our regression plot again, but let's add two new points on top:

* The predicted 2020 crash rate
* The actual 2020 crash rate

Code has been provided for the original regression plot and a legend title. 

Add a scatter plot layer to add the 2020 predicted and actual crash rates that both used the 2019 smartphone usage rate. Use different colors and marker shapes for the predicted and actual 2020 crash rates.


```python
# recreate the regression plot we made earlier
sns.regplot(x = 'Smartphone_usage', y = 'Crashes_per_100k', data = smartphones)


# add a scatter plot layer to show the actual and predicted 2020 values
sns.scatterplot(x=[81,81], y=[174.2047,157.88955], 
                hue=['predicted', 'actual'], 
                style=['predicted', 'actual'], 
                markers=['X', 'o'], 
                palette=['navy', 'orange'], 
                s=200)

# add legend title
plt.legend(title='2020')
plt.show()
```


    
![png](README_files/README_49_0.png)
    


<details>
    <summary style="display:list-item; font-size:16px; color:blue;"><i>What did we discover in this step? Toggle to check!</i></summary>

By adding another layer to our regression plot, we can see the difference between the predicted and real crash rates in February 2020. This allows us to see how these values compare to the rest of the dataset. 

</details>
