from decision_tree_classifier import DecisionTree_SkinLeasion_Classifier

clf = DecisionTree_SkinLeasion_Classifier(
    max_depth=5
)

clf.load_and_prepare("data/features.csv")

clf.train()

clf.evaluate()

clf.save_predictions("tree_predictions.csv")