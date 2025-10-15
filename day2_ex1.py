# Exercise 1: Amalyze a Dataset's Distribution
# url = "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv"

from scipy.stats import skew, kurtosis
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
# Load Dataset
url = "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv"
df = pd.read_csv(url)

# Analyse sepal_length
feature = df["sepal_length"]
print("Skewness: ", skew(feature))
print("Kurtosis: ", kurtosis(feature))

# Visualize distribution
sns.histplot(feature, kde=True)
plt.title("Distribution of Sepal Length")
plt.show()

# Additional Practice
# A. Compare the effects of skewness and kurtosis on different datasets.
# B. Simulate random variables from custom distributions.
# C. Explore datasets with real-world applications of distributions.

