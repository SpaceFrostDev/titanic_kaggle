# Kaggle Titanic Survival Prediction
## End Accuracy `83.33%`
Predicting passenger survival aboard the titanic in Kaggle's Titanic Dataset Competition. Below, I detail an executive summary of my process, including cleaning, encoding, splitting, normalizing, oversampling, model selection, training and inference.

## The Raw Data
![raw_df](./images/raw_df.png)  
Our prediction target is `Survived`.  

## Data Cleaning
Drop irrelevant columns: `cleaned_df = raw_df.drop(columns=['PassengerId', 'Ticket', 'Name'])`.  
![dropped_df](./images/dropped_df.png)  
You've noticed a few imperfections, such as `NaN`s. Let's see the extent of the problem.  
![count_nans](./images/count_nans.png)  
*NOTE*: The `cabin` variable would be very helpful to include. Undoubtedly, a passenger's position in the ship affects their mortality in the event of a disaster. However, 687 entries out of 891 are `NaN`, far too many blanks for this variable to be useful. So, we'll drop it aswell.
`cleaned_df.drop(columns=['Cabin'], inplace=True)`  

## Completing
The `Age` variable contains 177 `NaN`s, a small enough number to simply fill the empty with the average age. Alternatively, one could compute the distribution of the `Age` variable and fill the empty cells with numbers randomly selected from described distribution. `cleaned_df['Age'].fillna(cleaned_df['Age'].mean(), inplace=True)`.  

## Encoding
The `Sex` and `Embarked` variables need to be encoded from strings to numbers.  
```
cleaned_df['Sex'].replace({'male': 1, 'female': 0}, inplace=True)
cleaned_df['Embarked'].replace({'S': 0, 'C': 1, 'Q': 2, 'UNK': 3}, inplace=True)
```  
Our cleaned, encoded dataframe.  
![encoded_df](./images/encoded_df.png)  

## Splitting
```
X = cleaned_df.drop(columns=['Survived'])
y = cleaned_df['Survived']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
```  
## Normalization
After we split our data, we may scale our two continuous variables `Age` and `Fare`, fit our normalization `MinMaxScaler` to the training data.  
```
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(X_train[['Age', 'Fare']])

X_train_scaled[['Age', 'Fare']] = scaler.transform(X_train[['Age', 'Fare']])
```  
![normalized_df](./images/normalized_df.png)  

## Oversampling
Upon inspecting our dependant variable, `Survived`, we notice a distinct class imbalance.  
![imbalance](./images/imbalance.png)  
Let's correct that.  
```
oversampler = SMOTE()
X_train_scaled_os, y_train_os = oversampler.fit_resample(X_train_scaled, y_train)
```  
Much better :)  
![balance](./images/balance.png)  

## Model Creation
For this competition I'm going to elect to use a neural network as it's fairly robust to hyper=parameter choice and can be very easily implemented. My alternate choice would be to use a `RandomForest` model from the `scikit-learn` library, and iteratively search for the optimal hyper-parameter combination.  
*NOTE*: An entity embedding (in the form of `nn.Embedding`) is not needed in this model as our only categorical variable `Embarked` has a cardinality of four, and the rest of the variables, like `Pclass` are either ordinal, or continuous.  


## Training and Inference

## Technology Stack