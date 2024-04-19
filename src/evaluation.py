
class ModelSelector:
    """
    A class to compare multiple forecasting models based on their Mean Absolute Error
    (MAE). This is a PoC version, in a more advanced state the selection criteria should
    be configurable, as well as the data structure to store the models

    The models in the selector must follow (duck typing) the TimeSerieModel abc.
    """

    def __init__(self):
        self.models = []

    def add_model(self, model):
        self.models.append(model)

    def delete_model(self, model):
        if self.models and model in self.models:
            self.models.remove(model)

    def list_models(self):
        return self.models

    def compare_models(self):
        """
        Compares all models in the list by fitting them and evaluating their MAE.
        Returns the model with the lowest MAE.

        Returns:
            tuple: A tuple containing the best model and its MAE. If no models are
            present, returns (None, None).
        """
        min_mae = None
        best_model = None
        for model in self.models:
            model_mae = model.fit().mae
            if not min_mae or model_mae < min_mae:
                min_mae = model_mae
                best_model = model
        return best_model, min_mae

