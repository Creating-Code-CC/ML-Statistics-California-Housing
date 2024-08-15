import os
import tarfile
import urllib
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from six.moves import urllib

# FUNCTIONS TO FETCH DATA
DOWNLOAD_ROOT ="https://raw.githubusercontent.com/ageron/handson-ml/master/"
HOUSING_PATH = os.path.join("datasets", "housing")
HOUSING_URL = DOWNLOAD_ROOT + "datasets/housing/housing.tgz"

def fetch_housing_data(housing_url=HOUSING_URL, housing_path=HOUSING_PATH):
    if not os.path.isdir(housing_path):
        os.makedirs(housing_path)
    tgz_path = os.path.join(housing_path, "housing.tgz")
    urllib.request.urlretrieve(housing_url, tgz_path)
    housing_tgz = tarfile.open(tgz_path)
    housing_tgz.extractall(path=housing_path)
    housing_tgz.close()
    
def load_housing_data(housing_path=HOUSING_PATH):
    csv_path = os.path.join(housing_path, "housing.csv")
    return pd.read_csv(csv_path)
    
housing_files = fetch_housing_data()
housing = load_housing_data()
### EDA FAMILIARIZE YOURSELF WITH DATA SET
print(housing.info())
# Check the categories of ocean_proximity
counts = housing["ocean_proximity"].value_counts()
print(counts)
# Get numerical info
print(housing.describe())
# Get a visualization of the data. I like using seaborn pariplot
sns.pairplot(housing)
# You can also use a histogram to plot each feature
housing.hist(bins=50,figsize=(20,15))
plt.show()
# Split the data into Test and Train Sets (Good practice is to use 20% of the data for training.)
# We can check the margin of error against data the model hasnt't seen before, hence the test set.
from sklearn.model_selection import train_test_split
train_set, test_set = train_test_split(housing, test_size=.2, random_state=42)
#Stratified Sampling
housing["income_cat"] = np.ceil(housing["median_income"] / 1.5)
housing["income_cat"].where(housing["income_cat"] < 5, 5.0, inplace=True)
housing["income_cat"].hist()
plt.show()

from sklearn.model_selection import StratifiedShuffleSplit
model = StratifiedShuffleSplit(n_splits=1, test_size=.2, random_state=42)
for train_index, test_index in model.split(housing, housing["income_cat"]):
    strat_train_set = housing.loc[train_index]
    strat_test_set = housing.loc[test_index]
    
strat_test_set["income_cat"].value_counts() / len(strat_test_set)
# Drop so data is back in its original state
for set_ in (strat_train_set, strat_test_set):
    set_.drop("income_cat", axis=1, inplace=True)
# Make a copy of your data you wish to maniplulate (Never manipulate the original set)
housing = strat_train_set.copy()
# Trust but verify data is intact
housing.info()
# Visualize Geographical data with high density data points using multiple colors
housing.plot(kind="scatter", x='longitude', y='latitude', alpha=0.4,
s=housing["population"]/ 100, label = "population", figsize=(10,7),
c="median_house_value", cmap=plt.get_cmap("jet"), colorbar=True,
)
plt.legend()
# Finding correlations with seaborn heatmap
housing = housing.drop("ocean_proximity", axis=1)
corr_matrix = housing.corr()
sns.heatmap(corr_matrix, annot=True)
# Find correlation of a specific target
corr_matrix["median_house_value"].sort_values(ascending=False)
# Find correlation with Pandas
from pandas.plotting import scatter_matrix
attributes = ["median_house_value", "median_income","total_rooms", "housing_median_age"]
scatter_matrix(housing[attributes], figsize=(10,7))
# Plot the correlated attribute with a scatter plot (look for upward trend)
housing.plot(kind="scatter", x="median_income", y="median_house_value", alpha=.1)
plt.show()
# Some attributes are unintelligible at first glance like total_rooms and total_bedrooms. Before we start preparing our data for 
# algorithms, let's give more sense and insight for the two quantitive attributes
housing["rooms_per_household"] = housing["total_rooms"]/housing["households"]
housing["bedrooms_per_room"] = housing["total_bedrooms"]/housing["total_rooms"]
housing["population_per_household"] = housing["population"]/housing["households"]
# Trust but verify features are intact
housing.info()
# Peek into the data (bedrooms_per_room seems off)
housing.head()
# See if new correlations appear
corr_matrix = housing.corr()
corr_matrix["median_house_value"].sort_values(ascending=False)
# PREPARE THE DATA || separate the predictors from labels
# once again, do this with a CLEAN data set, make a copy of strat_train_set
housing = strat_train_set.drop("median_house_value", axis=1)
housing_labels = strat_train_set["median_house_value"].copy()
housing.info()

# DATA CLEANING 
# 3 options for taking care of missing data (drop, dropna, fillna) 
housing = housing.dropna(subset=["total_bedrooms"])
# housing = housing.drop("total_bedrooms", axis=1)
# median = housing["total_bedrooms"].median()
# housing["total_bedrooms"].fillna(median, inplace=True)
# housing.info()

# option 3 should be done to the train set and the values computed should be saved when evaluating your system with the test set

# DATA CLEANING CONTIN'D (imputer) SECTION COMMENTED OUT AS SYSTEM CANNOT IMPORT IMPUTER, LEFT HERE FOR REFERENCE
# strategy is known as a hyperparameter
# from sklearn.preprocessing import Imputer 
# imputer = Imputer(strategy="median")
# Since the median can only be computed on numerical attributes, we need to drop all non numeric attributes first

# housing_num = housing.drop("ocean_proximity", axis=1)
# imputer.fit(housing_num)

# The results of the imputer can be found in imputer.statistics or housing_num.median().values()
# imputer.statistics_ 
# housing_num.median().values()
# IMPUTER CONTIN'D
# Now we can use the "trained" imputer to transform the data set by replacing missing values with the learned medians
# The output will be a plain Numpy Array
# X = imputer.transform(housing_num)
# Putting the imputer results into a DataFrame
# housing_tr = pd.DataFrame(X, columns=housing_num.columns)


