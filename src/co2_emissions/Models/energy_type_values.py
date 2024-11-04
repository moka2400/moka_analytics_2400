class EnergyTypeValues:
    all_energy_types = 'all_energy_types'
    coal = 'coal'
    natural_gas = 'natural_gas'
    petroleum_n_other_liquids = 'petroleum_n_other_liquids'
    nuclear = 'nuclear'
    renewables_n_other = 'renewables_n_other'

    @classmethod
    def get_energy_types(cls):
        return [value for name, value in vars(cls).items() 
                if not name.startswith('__') and 
                not callable(getattr(cls, name))]