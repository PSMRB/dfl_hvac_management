from pandas import Series as pdSeries


class Market:
    """
        Market is a class which contains the necessary information about a distribution electricity market.
    """
    def __init__(self, demand_charge: float, prices_import: pdSeries, prices_export: pdSeries,
                 line_capacity: float = 1e6):
        """
        Initialize the market
        :param demand_charge: a float which is the demand charge €/kW
        :param prices_import: a time series which is the prices of the electricity import €/kWh. The index of the
                pandas.Series is the datetime of the prices. It must match those of the EMS
        :param prices_export: a time series which is the prices of the electricity export €/kWh. The index of the
                pandas.Series is the datetime of the prices. It must match those of the EMS
        :param line_capacity: a float which is the capacity of the line in kW
        """
        self.demand_charge = demand_charge
        self.prices_import = prices_import
        self.prices_export = prices_export
        self.line_capacity = line_capacity

