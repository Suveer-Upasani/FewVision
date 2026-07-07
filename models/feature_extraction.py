# models/feature_extraction.py
"""Feature extraction — redirect stub.

The live feature extraction implementation lives in
``modules/feature_extraction/``.  This file is kept for historical
reference only and does **not** contain runnable code.

Implemented extractors
----------------------
+------------+--------------------------------------------------+---------+
| Identifier | Class                                            | Dim     |
+============+==================================================+=========+
| dinov2     | modules.feature_extraction.dinov2_extractor      | 384–    |
|            |   .DINOv2Extractor                               | 1536    |
+------------+--------------------------------------------------+---------+
| vit        | modules.feature_extraction.vit_extractor         | 768     |
|            |   .ViTExtractor                                  |         |
+------------+--------------------------------------------------+---------+
| resnet50   | modules.feature_extraction.resnet_extractor      | 2048    |
|            |   .ResNet50Extractor                             |         |
+------------+--------------------------------------------------+---------+

Usage (all pipeline code)
-------------------------
    from modules.feature_extraction import get_extractor

    extractor = get_extractor()          # reads config.FEATURE_EXTRACTOR
    extractor = get_extractor("resnet50")  # explicit name
    extractor = get_extractor("dinov2", model_variant="dinov2_vitb14")

    extractor.load_model()               # downloads weights on first run
    embedding = extractor.extract(bgr_image)  # np.ndarray shape (D,)

Switching extractors
--------------------
Change ONE line in ``config.py``:

    FEATURE_EXTRACTOR = "dinov2"    # self-supervised ViT, 384-dim (default)
    FEATURE_EXTRACTOR = "vit"       # supervised ViT-B/16, 768-dim
    FEATURE_EXTRACTOR = "resnet50"  # supervised ResNet50, 2048-dim

Or set the environment variable without changing any file:

    $env:FEWVISION_EXTRACTOR = "resnet50"; python app.py

No pipeline code changes are ever needed when switching extractors.
"""
