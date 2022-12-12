# Forest Fire Area Predictions

import pandas
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import norm
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet
from sklearn.preprocessing import PolynomialFeatures, MinMaxScaler, LabelEncoder
from sklearn.svm import SVR
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
import warnings

warnings.simplefilter('ignore')


def main():
    # import the dataset
    forest = pandas.read_csv("forestfires.csv")

    # Reducing the Right Skewness of the Area using log(n) + 1
    forest['u_area'] = np.log(forest['area'] + 1)

    data = norm.rvs(forest['area'])
    # Fit a normal distribution
    mu, std = norm.fit(data)
    plt.hist(data, bins=25, density=True, alpha=0.6, color='g')
    # Plot the Chart
    xmin, xmax = plt.xlim()
    x = np.linspace(xmin, xmax, 100)
    p = norm.pdf(x, mu, std)
    plt.plot(x, p, 'k', linewidth=2)
    title = "Results: population mean = %.2f,  Standard deviation = %.2f" % (mu, std)
    plt.title(title)

    # chart after reducing Right Skewness
    # Set Parameters for Chart
    plt.rcParams['figure.figsize'] = [20, 10]
    sns.set(style='white', font_scale=1)
    fig, ax = plt.subplots(1, 2)
    #  Distribution Plots
    area = sns.distplot(forest['area'], ax=ax[0], color='green')
    area2 = sns.distplot(forest['u_area'], ax=ax[1], color='green')
    area.set(title="Area Distribution", xlabel="Area", ylabel="Density")
    area2.set(title="Area Distribution After Reduced Skewness", xlabel="U_Area", ylabel="Density")

    # correlation heatmap
    plt.rcParams['figure.figsize'] = [12, 10]
    sns.set(font_scale=1)
    sns.heatmap(forest.corr(), cmap="YlGnBu", annot=True);
    # plt.show()

    # From DATA PREPROCESSING
    # Reducing the skewness for training and drop variable: u_area
    forest['area'] = np.log(forest['area'] + 1)
    forest.drop(columns='u_area', inplace=True)
    # print(forest)

    # Normalization of the features
    minmax = MinMaxScaler()

    # ISI, SC, RH, FFMC, DMC
    forest['ISI'] = minmax.fit_transform(np.array(forest['ISI']).reshape(-1, 1))
    forest['DC'] = minmax.fit_transform(np.array(forest['DC']).reshape(-1, 1))
    forest['RH'] = minmax.fit_transform(np.array(forest['RH']).reshape(-1, 1))
    forest['FFMC'] = minmax.fit_transform(np.array(forest['FFMC']).reshape(-1, 1))
    forest['DMC'] = minmax.fit_transform(np.array(forest['DMC']).reshape(-1, 1))

    # Changing letters into numerical values
    # change Months to numerical values
    forest['month'].replace({'jan': 1, 'feb': 2, 'mar': 3, 'apr': 4, 'may': 5, 'jun': 6,
                             'jul': 7, 'aug': 8, 'sep': 9, 'oct': 10, 'nov': 11, 'dec': 12},
                            inplace=True)

    # change Days to numerical values
    forest['day'].replace({'sun': 1, 'mon': 2, 'tue': 3, 'wed': 4, 'thu': 5, 'fri': 6, 'sat': 7}, inplace=True)

    # Split dataset into training and testing
    target = forest['area']
    features = forest.drop(columns='area')
    # set testing and training as 20-80
    x_train, x_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=196)
    print("Training data set size : ", x_train.shape)
    print("Testing data set size : ", x_test.shape)

    # Linear Regression Model
    model = LinearRegression()
    model.fit(x_train, y_train)

    # Predictions
    predictions = model.predict(x_test)
    print(predictions)

    # MSE: The smaller the mean squared error, the closer you are to finding the line of best fit
    print("Mean Squared Error : ", mean_squared_error(y_test, predictions))
    # Coefficient of determination
    print("Coefficient of determination : ", r2_score(y_test, predictions))


main()
