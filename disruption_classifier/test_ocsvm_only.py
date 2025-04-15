from matplotlib.pylab import f
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

from classifier import DisruptionClassifier, DISCHARGE_DATA_PATH, DISCHARGES_ETIQUETATION_DATA_FILE
from ocsvm_model import prepare_features_for_ocsvm, train_ocsvm_model, predict_with_ocsvm

def run_ocsvm_test(data_path, discharge_list_path, test_size=0.3, random_state=42, ocsvm_params=None):
    """
    Loads data, trains, and evaluates only the OCSVM model.
    """
    print("--- Starting OCSVM Only Test ---")

    # 1. Initialize Classifier and Load Data
    classifier = DisruptionClassifier(data_path, discharge_list_path)
    print("Loading data...")
    classifier.load_all_data()
    print("Data loading complete.")

    # 2. Prepare OCSVM Features
    print("Preparing OCSVM features...")
    X_ocsvm, y_ocsvm, ids_ocsvm, _ = prepare_features_for_ocsvm(
        classifier.discharge_data, classifier.discharge_labels
    )
    print(f"Feature preparation complete. Shape: {X_ocsvm.shape}")

    # 3. Split Data
    print(f"Splitting data (Test size: {test_size}, Random state: {random_state})...")
    X_train, X_test, y_train, y_test, ids_train, ids_test = train_test_split(
        X_ocsvm, y_ocsvm, ids_ocsvm,
        test_size=test_size,
        random_state=random_state,
        stratify=y_ocsvm  # Stratify to maintain class proportion
    )
    print(f"Train set size: {X_train.shape[0]}, Test set size: {X_test.shape[0]}")
    print(f"Train labels distribution: {np.bincount(y_train)}")
    print(f"Test labels distribution: {np.bincount(y_test)}")


    # 4. Scale Features
    print("Scaling features using StandardScaler...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    print("Scaling complete.")

    # 5. Train OCSVM Model
    print("Training OCSVM model...")
    # Note: train_ocsvm_model internally filters for y_train == 0
    ocsvm_model = train_ocsvm_model(X_train_scaled, y_train, params=ocsvm_params)
    print("Model training complete.")

    # Print support vector information
    print("\n--- Support Vector Analysis ---")
    n_support_vectors = ocsvm_model.support_vectors_.shape[0]
    n_training_samples = np.sum(y_train == 0)
    print(f"Number of support vectors: {n_support_vectors} ({n_support_vectors/n_training_samples:.2%} of training data)")
    
    # Print statistics about support vectors
    sv_mean = np.mean(ocsvm_model.support_vectors_, axis=0)
    sv_std = np.std(ocsvm_model.support_vectors_, axis=0)
    print(f"Support vector mean: {np.mean(sv_mean):.4f}")
    print(f"Support vector std: {np.mean(sv_std):.4f}")
    
    # Print decision function scores
    train_scores = ocsvm_model.decision_function(X_train_scaled)
    test_scores = ocsvm_model.decision_function(X_test_scaled)
    print(f"Training set decision scores: min={np.min(train_scores):.4f}, mean={np.mean(train_scores):.4f}, max={np.max(train_scores):.4f}")
    print(f"Test set decision scores: min={np.min(test_scores):.4f}, mean={np.mean(test_scores):.4f}, max={np.max(test_scores):.4f}")

    # 6. Predict on Test Set
    print("Predicting on the test set...")
    # Note: predict_with_ocsvm maps OCSVM output (-1, 1) to (1, 0)
    y_pred = predict_with_ocsvm(ocsvm_model, X_test_scaled)
    print("Prediction complete.")

    # 7. Evaluate
    print("\n--- OCSVM Evaluation Results ---")
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=['Normal (0)', 'Anomaly (1)'], zero_division=0)
    cm = confusion_matrix(y_test, y_pred)

    print(f"Accuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(report)
    print("\nConfusion Matrix:")
    print(cm)
    print("---------------------------------")

    # Optional: Analyze misclassifications
    print("\nAnalyzing misclassifications (Test Set):")
    misclassified_indices = np.where(y_test != y_pred)[0]
    for i in misclassified_indices:
        discharge_id = ids_test[i]
        print(f"  Discharge ID: {discharge_id}, Actual: {y_test[i]}, Predicted: {y_pred[i]}")

    return cm


if __name__ == "__main__":
    test_nu = [0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9]
    # test_gamma = ['scale', 'auto']
    test_gamma = [0.001, 0.01, 0.1, 1.0, 4.21279962787536]
    # test_kernel = ['linear', 'poly', 'rbf', 'sigmoid']
    test_kernel = ['rbf']
    
    iterations = 0
    total_iterations = len(test_nu) * len(test_gamma) * len(test_kernel)

    confussion_matrices = []

    for nu in test_nu:
        for gamma in test_gamma:
            for kernel in test_kernel:
                print(f"Testing OCSVM with nu={nu}, gamma={gamma}, kernel={kernel}")
                ocsvm_test_params = {'kernel': kernel, 'nu': nu, 'gamma': gamma}
                cm = run_ocsvm_test(
                    data_path=DISCHARGE_DATA_PATH,
                    discharge_list_path=DISCHARGES_ETIQUETATION_DATA_FILE,
                    test_size=0.3, # Use the same test size as in the main classifier script
                    random_state=42, # Use the same random state for consistency
                    ocsvm_params=ocsvm_test_params
                )
                confussion_matrices.append(cm)
                iterations += 1
                print(f"Completed test {iterations}/{total_iterations}")


    print("\n--- All Tests Completed ---")
    print("="*50)
    print("Confusion Matrices for all tests:")
    for i, cm in enumerate(confussion_matrices):
        print(f"Test {i+1}:")
        print(cm)
        print("="*50)
    