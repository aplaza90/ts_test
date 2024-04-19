from data_obtention import read_data, store_data
from models import RollingMa, SarimaxModelAdapter
from evaluation import ModelSelector
from data_exploration import plot_preds


def main():
    # Read and preprocess data
    data = read_data('train.csv')

    # Initialize models
    model_1 = SarimaxModelAdapter(data)
    model_2 = RollingMa(data)

    # Model selection
    ms = ModelSelector()
    ms.add_model(model_1)
    ms.add_model(model_2)
    best_model, min_mae = ms.compare_models()

    # Forecasting and plotting
    predictions = best_model.fit().forecast(12)
    plot_preds(data, predictions)

    # Predictions to csv
    store_data(predictions, 'predictions.csv')


if __name__ == "__main__":
    main()
