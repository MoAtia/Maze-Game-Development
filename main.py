from data.loader import load_data, get_data_shape
from data.preprocessing import preprocess_data
from data.visualization import plot_sample
from models.train import train_models
from models.evaluate import evaluate_model
from models.save_load import save_model
from utils.seeds import set_seeds
from config import TEST_SIZE, RANDOM_STATE


def main():
    # Set random seeds
    set_seeds()
    
    # Load data
    df = load_data()
    print("Data shape:", get_data_shape(df))
    
    # Visualize samples
    sample = df.sample(5, random_state=RANDOM_STATE)
    plot_sample(sample)
    
    # Preprocess data
    X_train, X_test, y_train, y_test, encoder = preprocess_data(df, test_size=TEST_SIZE, random_state=RANDOM_STATE)
    
    # Train models
    trained_models = train_models(X_train, y_train)
    
    # Evaluate models
    for name, model in trained_models.items():
        print(f"\nEvaluating {name}:")
        evaluate_model(model, X_test, y_test)
        save_model(model, name)

if __name__ == "__main__":
    main()