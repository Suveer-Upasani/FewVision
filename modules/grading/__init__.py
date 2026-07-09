# modules/grading/__init__.py
"""Product Grading Core module for FewVision.

Exposes the ProductGrader service and the ProductGrade result model.
"""

from modules.grading.grading_result import ProductGrade
from modules.grading.product_grader import ProductGrader

__all__ = ["ProductGrader", "ProductGrade"]
