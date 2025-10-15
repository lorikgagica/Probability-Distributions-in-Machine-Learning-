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
from scipy.stats import skew, kurtosis
import numpy as np

# Dataset 1: Normal distribution
data1 = np.random.normal(0, 1, 1000)
# Dataset 2: Exponential (skewed)
data2 = np.random.exponential(1, 1000)

print("Normal dataset - Skew:", round(skew(data1), 2), "Kurtosis:", round(kurtosis(data1), 2))
print("Exponential dataset - Skew:", round(skew(data2), 2), "Kurtosis:", round(kurtosis(data2), 2))


# B. Simulate random variables from custom distributions.
import numpy as np

# Simulate 10 values from a custom uniform distribution [5, 10]
uniform_custom = np.random.uniform(5, 10, 10)
print("Uniform [5,10]:", uniform_custom)

# Simulate 10 values from a binomial distribution (n=5, p=0.3)
binomial_custom = np.random.binomial(5, 0.3, 10)
print("Binomial (n=5, p=0.3):", binomial_custom)


# C. Explore datasets with real-world applications of distributions.
import numpy as np

# Simulate daily steps (could fit a normal distribution)
daily_steps = np.random.normal(8000, 1500, 7)  # 7 days
print("Simulated daily steps:", np.round(daily_steps))

# Simulate number of emails received per day (Poisson: event counts)
daily_emails = np.random.poisson(15, 7)
print("Simulated daily emails:", daily_emails)
