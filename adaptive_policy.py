"""Simple adaptive policy for deciding augmentations based on quality recommendations.

For this prototype we simply forward the recommendations from the quality checker.
More sophisticated logic could consider content metrics as well.
"""

class AdaptivePolicy:
    @staticmethod
    def decide_augmentations(quality_dict: dict, content_dict: dict) -> list:
        """Return a list of augmentation names.

        Currently just returns the `recommendations` field from the quality dict.
        """
        return quality_dict.get("recommendations", [])
