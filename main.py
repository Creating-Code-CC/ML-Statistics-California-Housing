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


