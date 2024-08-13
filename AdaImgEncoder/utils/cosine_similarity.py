import numpy as np
import evaluate
import datasets
_DESCRIPTION = """
Cosine similarity is a measure of similarity between two non-zero vectors of an inner product space that measures the cosine of the angle between them.
The cosine of 0° is 1, and it is less than 1 for any other angle.
It is thus a judgment of orientation and not magnitude: two vectors with the same orientation have a cosine similarity of 1, two vectors at 90° have a similarity of 0, and two vectors diametrically opposed have a similarity of -1, independent of their magnitude.
"""


_KWARGS_DESCRIPTION = """
Args:
    predictions: list of list of float or float, list of predictions to score.
    references: list of list of float or float, list of references to score against.
Returns:
    cosine_similarity: float, cosine similarity between predictions and references.
    
"""
class CosineSimilarity(evaluate.Metric):
    def _info(self):
        return evaluate.MetricInfo(
            description=_DESCRIPTION,
            citation="",
            inputs_description=_KWARGS_DESCRIPTION,
            features=datasets.Features(
                {
                    "predictions": 
                        datasets.Sequence(datasets.Value("float32")),
                    "references":
                        datasets.Sequence(datasets.Value("float32")),
                }
            ),
            reference_urls="",
            
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
