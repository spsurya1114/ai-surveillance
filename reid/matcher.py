import numpy as np

class Matcher:
    def __init__(self, threshold=0.6):
        self.database = {}
        self.global_id = 0
        self.threshold = threshold

    def cosine_similarity(self, a, b):
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

    def match(self, feature):
        best_match = None
        best_score = 0

        for gid, stored_feat in self.database.items():
            sim = self.cosine_similarity(feature, stored_feat)

            if sim > best_score:
                best_score = sim
                best_match = gid

        if best_score > self.threshold:
            return best_match

        # New person
        self.global_id += 1
        self.database[self.global_id] = feature
        return self.global_id
