from abc import ABC, abstractmethod

from statsmodels.tsa.statespace import sarimax
from statsmodels.tsa.seasonal import seasonal_decompose
from pmdarima import auto_arima


class TimeSeriesModel(ABC):
    """
    Abstract base class for time series forecasting models.
    """

    def __init__(self, timeseries):
        self.timeseries = timeseries

    @abstractmethod
    def fit(self):
        pass

    @abstractmethod
    def forecast(self, n_of_values):
        """
        Forecasts future values of the time series.

        Args:
            n_of_values (int): Number of future values to forecast.

        Returns:
            pd.Series: Predicted values of the time series.
        """
        pass

    @property
    @abstractmethod
    def mae(self):
        """
        Calculates the Mean Absolute Error of the model residuals.

        Returns:
            float: The Mean Absolute Error.
        """
        pass


class RollingMa(TimeSeriesModel):
    """
    Implements a rolling moving average forecasting model using seasonal decomposition.
    """

    def __init__(self, timeseries):
        super().__init__(timeseries)
        try:
            self.seasonal_decomp = seasonal_decompose(timeseries, two_sided=False)
        except Exception as e:
            raise ValueError(f"Failed to decompose series: {str(e)}")

    def fit(self):
        return self

    def forecast(self, n_of_values):
        """
        Forecasts future values based on the moving average of the last year and
        the seasonal factors derived from the seasonal decomposition.
        """
        month = self.seasonal_decomp.seasonal.index.month
        seasonal_factor = self.seasonal_decomp.seasonal.groupby(month).mean()

        predictions = self.timeseries.tail(12)
        for i in range(n_of_values):
            next_index = predictions.index.shift(1)[-1]
            next_value = (
                    predictions.tail(12).mean() +
                    seasonal_factor.loc[next_index.month]
            )
            predictions[next_index] = next_value

        return predictions[12:]

    @property
    def mae(self):
        return self.seasonal_decomp.resid.abs().mean()


class SarimaxModelAdapter(TimeSeriesModel):
    """
    Adapter class for the SARIMAX model, fitting the model to time series data
    and providing forecasting capabilities.

    The SARIMAX is fitted based on automatic model order selection using AIC
    through the pmdarima package's auto_arima function.
    """
    def __init__(self, timeseries):
        """
        Initializes the SarimaxModelAdapter with time series data and constructs
        the SARIMAX model using automatically determined order parameters.
        """
        super().__init__(timeseries)
        self.model = self.sarimax_factory()

    def fit(self):
        self.model = self.model.fit()
        return self

    def forecast(self, n_of_values):
        return self.model.forecast(n_of_values)

    @property
    def mae(self):
        return self.model.mae

    @staticmethod
    def sarimax_selector(series_data):
        """
        Selects the optimal SARIMAX model order using the auto_arima function from
        pmdarima, based on the Akaike Information Criterion (AIC).

        Args:
            series_data (pd.Series): The time series data for which to determine the
            model order.

        Returns:
            tuple: A tuple containing the (order, seasonal_order) of the SARIMAX model.
        """
        sarimax_model = auto_arima(
            series_data,
            start_p=0,
            start_q=0,
            max_p=3,
            max_q=3,
            m=12,
            test='adf',
            seasonal=True,
            trace=True
        )
        return sarimax_model.order, sarimax_model.seasonal_order

    def sarimax_factory(self):
        """
        Constructs the SARIMAX model using the self.timeseries and the params selected
        by the self.sarimax_selector.

        Returns:
            SARIMAX: An instance of the SARIMAX model prepared for fitting.
        """
        order, seasonal_order = self.sarimax_selector(self.timeseries)
        return sarimax.SARIMAX(
            self.timeseries, order=order, seasonal_order=seasonal_order
        )