import numpy as np
import evaluate

class CosineSimilarity(evaluate.Metric):
    def _info(self):
        return evaluate.MetricInfo(
            description="Cosine similarity between two vectors.",
            citation="No formal citation.",
            features=["predictions", "references"],
        )

    def _compute(self, predictions, references):
        if len(predictions) != len(references):
            raise ValueError("predictions and references must have the same length")

        similarities = []
        for p, r in zip(predictions, references):
            p = np.asarray(p)
            r = np.asarray(r)
            similarity = np.dot(p, r) / (np.linalg.norm(p) * np.linalg.norm(r))
            similarities.append(similarity)

        return {"cosine_similarity": np.mean(similarities)}
