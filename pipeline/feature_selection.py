class FeatureSelection:
    def __init__(self,X_set,y_set):
        self.X_set = X_set
        self.y_set = y_set
# Selecting important only features for training with threshold=0.1
    def select_features(self):
        threshold=0.1
        target_corr = self.X_set.corrwith(self.y_set, numeric_only=True)
        selected = target_corr[abs(target_corr) > threshold].index.tolist()
        for feature in selected:
            print(feature, ":", target_corr[feature])
        return selected
