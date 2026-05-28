import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent / "results" / "models"))
from Knn_classifier import KNNSkinCancerClassifier
from decision_tree_classifier import DecisionTree_SkinLeasion_Classifier

def main(features_path, knn_prediction_results_path, decisiontree_prediction_results_path,
         knn_model_path, decisiontree_model_path, load_model):
    """
    Docstring for main
    
    :param features_path: Path to the features csv used as input to the model (e.g. ./data/features.csv).
    :param knn_prediction_results_path: Path to save KNN predictions (e.g. ./results/predictions/predictions_knn.csv). # RENAMED FROM prediction_results_path, NOW SPECIFIC TO KNN
    :param decisiontree_prediction_results_path: Path to save Decision Tree predictions (e.g. ./results/predictions/predictions_decisiontree.csv). # ADDED NEW PARAM FOR DECISION TREE
    :param knn_model_path: Path to save or load the KNN model (e.g. ./results/models/knn_model.pkl). # RENAMED FROM model_path, NOW SPECIFIC TO KNN
    :param decisiontree_model_path: Path to save or load the Decision Tree model (e.g. ./results/models/decisiontree_model.pkl). # ADDED NEW PARAM FOR DECISION TREE
    :param load_model: Boolean to train the model and save it to model_path if False, load it from model_path if True. 
    """
    knn = KNNSkinCancerClassifier(n_neighbors=43, weights='distance', metric='manhattan')
    clf = DecisionTree_SkinLeasion_Classifier(max_depth=5)

    # load dataset CSV file
    # split the dataset into training and testing sets.
    knn.load_and_prepare(features_path)
    clf.load_and_prepare(features_path)

    if load_model:
        # load the model
        knn.load_model(knn_model_path)
        clf.load_model(decisiontree_model_path)
        
    else:
        # train the classifier (using logistic regression as an example)
        knn.train()
        clf.train()
        # save the model.
        knn.save_model(knn_model_path)
        clf.save_model(decisiontree_model_path)

    # test the classifier.
    knn.evaluate()
    clf.evaluate()

    # write test results to CSV.
    knn.save_predictions(knn_prediction_results_path)
    clf.save_predictions(decisiontree_prediction_results_path)


if __name__ == "__main__":
    base = Path(__file__).parent  

    features_path                        = base / "data" / "features.csv"
    knn_prediction_results_path          = base / "results" / "predictions" / "predictions_knn.csv"
    decisiontree_prediction_results_path = base / "results" / "predictions" / "predictions_decisiontree.csv"
    knn_model_path                       = base / "results" / "models" / "knn_model.pkl"       
    decisiontree_model_path              = base / "results" / "models" / "decisiontree_model.pkl"  
    load_model = False

   
    main(features_path, knn_prediction_results_path, decisiontree_prediction_results_path,
         knn_model_path, decisiontree_model_path, load_model)