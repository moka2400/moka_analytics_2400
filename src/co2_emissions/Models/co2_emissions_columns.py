class EnergyColumns:
    Country = 'Country'
    EnergyType = 'Energy_type'
    Year = 'Year'
    EnergyConsumption = 'Energy_consumption'
    EnergyProduction = 'Energy_production'
    GDP = 'GDP'
    Population = 'Population'
    EnergyIntensityPerCapita = 'Energy_intensity_per_capita'
    EnergyIntensityByGDP = 'Energy_intensity_by_GDP'
    CO2Emission = 'CO2_emission'

    @classmethod
    def get_energy_columns(cls):
        return [
            cls.Country,
            cls.EnergyType,
            cls.Year,
            cls.Consumption,
            cls.Production,
            cls.GDP,
            cls.Population,
            cls.IntensityPerCapita,
            cls.IntensityByGDP,
            cls.CO2Emission 
        ]