from co2_emissions import load_data, EnergyColumns, EnergyTypeValues, CountryValues
import pandas as pd
import matplotlib.pyplot as plt

import statsmodels.api as sm
import seaborn as sns
import matplotlib.pyplot as plt
from statsmodels.regression.linear_model import RegressionResultsWrapper

def scatterplot(x_axis: str, y_axis: str, data: pd.DataFrame):
    sns.scatterplot(x=x_axis, y=y_axis, data=data, color='blue', s=100)

    # plt.figure(figsize=(12, 8))

    plt.title('Coal usage')
    plt.xlabel('Year')
    plt.ylabel('CO2 consumption')
    plt.grid()
    plt.tight_layout()
    plt.show()

def run_eda():
    energytypeColumn = EnergyColumns.EnergyType
    countryColumn = EnergyColumns.Country
    emissionColumn = EnergyColumns.CO2Emission
    yearColumn = EnergyColumns.Year

    data = load_data()

    denmark = CountryValues.Denmark
    coal = EnergyTypeValues.coal
    denmark_coal = data.loc[(data[energytypeColumn] == coal) & (data[countryColumn] == denmark)]

    year = denmark_coal[yearColumn]
    co2_emission = denmark_coal[emissionColumn]

    print(denmark_coal)
    scatterplot(year, co2_emission, denmark_coal)

if __name__ == "__main__":
    run_eda()
    