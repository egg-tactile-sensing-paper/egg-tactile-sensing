# Cross-validated detection results

56 finished runs in `cross_validation/results`.

## Grouped CV (pretrained)

| Model | Precision | Recall | mAP50 | mAP50:95 | folds |
| --- | --- | --- | --- | --- | --- |
| YOLOv8m | 0.850 ± 0.073 | 0.862 ± 0.029 | 0.898 ± 0.035 | 0.676 ± 0.030 | 5 |
| YOLOv9m | 0.832 ± 0.050 | 0.854 ± 0.025 | 0.878 ± 0.038 | 0.637 ± 0.070 | 5 |
| YOLOv10m | 0.861 ± 0.067 | 0.843 ± 0.042 | 0.874 ± 0.047 | 0.646 ± 0.054 | 5 |
| YOLOv11m | 0.846 ± 0.071 | 0.850 ± 0.030 | 0.868 ± 0.038 | 0.640 ± 0.047 | 5 |

Per fold (held-out eggs only):

| Model | fold | test images | Precision | Recall | mAP50 | mAP50:95 |
| --- | --- | --- | --- | --- | --- | --- |
| YOLOv8m | 0 | 83 | 0.882 | 0.834 | 0.878 | 0.660 |
| YOLOv8m | 1 | 84 | 0.889 | 0.881 | 0.936 | 0.684 |
| YOLOv8m | 2 | 72 | 0.915 | 0.830 | 0.908 | 0.641 |
| YOLOv8m | 3 | 75 | 0.831 | 0.897 | 0.922 | 0.723 |
| YOLOv8m | 4 | 63 | 0.731 | 0.870 | 0.849 | 0.674 |
| YOLOv9m | 0 | 83 | 0.855 | 0.849 | 0.889 | 0.648 |
| YOLOv9m | 1 | 84 | 0.880 | 0.881 | 0.921 | 0.650 |
| YOLOv9m | 2 | 72 | 0.800 | 0.822 | 0.822 | 0.522 |
| YOLOv9m | 3 | 75 | 0.762 | 0.876 | 0.865 | 0.655 |
| YOLOv9m | 4 | 63 | 0.863 | 0.843 | 0.895 | 0.712 |
| YOLOv10m | 0 | 83 | 0.831 | 0.812 | 0.848 | 0.579 |
| YOLOv10m | 1 | 84 | 0.920 | 0.911 | 0.924 | 0.697 |
| YOLOv10m | 2 | 72 | 0.932 | 0.844 | 0.907 | 0.651 |
| YOLOv10m | 3 | 75 | 0.768 | 0.804 | 0.808 | 0.605 |
| YOLOv10m | 4 | 63 | 0.854 | 0.845 | 0.883 | 0.699 |
| YOLOv11m | 0 | 83 | 0.893 | 0.843 | 0.878 | 0.668 |
| YOLOv11m | 1 | 84 | 0.881 | 0.865 | 0.921 | 0.694 |
| YOLOv11m | 2 | 72 | 0.902 | 0.808 | 0.877 | 0.570 |
| YOLOv11m | 3 | 75 | 0.731 | 0.888 | 0.842 | 0.626 |
| YOLOv11m | 4 | 63 | 0.823 | 0.846 | 0.821 | 0.644 |

Pooled over the folds (all 19 eggs, each scored by a model that never saw it):

| Model | pooled mAP50 | pooled mAP50:95 |
| --- | --- | --- |
| YOLOv8m | 0.849 | 0.635 |
| YOLOv9m | 0.812 | 0.594 |
| YOLOv10m | 0.819 | 0.598 |
| YOLOv11m | 0.821 | 0.601 |

## Grouped CV (from scratch)

| Model | Precision | Recall | mAP50 | mAP50:95 | folds |
| --- | --- | --- | --- | --- | --- |
| YOLOv8m | 0.802 ± 0.066 | 0.844 ± 0.071 | 0.877 ± 0.054 | 0.644 ± 0.066 | 5 |
| YOLOv9m | 0.803 ± 0.048 | 0.811 ± 0.052 | 0.848 ± 0.023 | 0.618 ± 0.029 | 5 |
| YOLOv10m | 0.783 ± 0.064 | 0.840 ± 0.035 | 0.837 ± 0.073 | 0.608 ± 0.064 | 5 |
| YOLOv11m | 0.859 ± 0.025 | 0.843 ± 0.037 | 0.883 ± 0.025 | 0.661 ± 0.043 | 5 |

Per fold (held-out eggs only):

| Model | fold | test images | Precision | Recall | mAP50 | mAP50:95 |
| --- | --- | --- | --- | --- | --- | --- |
| YOLOv8m | 0 | 83 | 0.767 | 0.865 | 0.859 | 0.631 |
| YOLOv8m | 1 | 84 | 0.844 | 0.942 | 0.949 | 0.716 |
| YOLOv8m | 2 | 72 | 0.852 | 0.785 | 0.850 | 0.565 |
| YOLOv8m | 3 | 75 | 0.703 | 0.766 | 0.813 | 0.603 |
| YOLOv8m | 4 | 63 | 0.846 | 0.865 | 0.912 | 0.706 |
| YOLOv9m | 0 | 83 | 0.812 | 0.740 | 0.815 | 0.602 |
| YOLOv9m | 1 | 84 | 0.784 | 0.845 | 0.867 | 0.652 |
| YOLOv9m | 2 | 72 | 0.868 | 0.832 | 0.872 | 0.595 |
| YOLOv9m | 3 | 75 | 0.812 | 0.774 | 0.837 | 0.595 |
| YOLOv9m | 4 | 63 | 0.738 | 0.863 | 0.850 | 0.647 |
| YOLOv10m | 0 | 83 | 0.790 | 0.826 | 0.788 | 0.568 |
| YOLOv10m | 1 | 84 | 0.873 | 0.838 | 0.930 | 0.673 |
| YOLOv10m | 2 | 72 | 0.696 | 0.901 | 0.856 | 0.562 |
| YOLOv10m | 3 | 75 | 0.760 | 0.818 | 0.741 | 0.555 |
| YOLOv10m | 4 | 63 | 0.793 | 0.815 | 0.868 | 0.684 |
| YOLOv11m | 0 | 83 | 0.892 | 0.854 | 0.888 | 0.662 |
| YOLOv11m | 1 | 84 | 0.832 | 0.886 | 0.912 | 0.703 |
| YOLOv11m | 2 | 72 | 0.858 | 0.834 | 0.856 | 0.598 |
| YOLOv11m | 3 | 75 | 0.840 | 0.785 | 0.860 | 0.644 |
| YOLOv11m | 4 | 63 | 0.876 | 0.854 | 0.901 | 0.699 |

Pooled over the folds (all 19 eggs, each scored by a model that never saw it):

| Model | pooled mAP50 | pooled mAP50:95 |
| --- | --- | --- |
| YOLOv8m | 0.771 | 0.559 |
| YOLOv9m | 0.771 | 0.552 |
| YOLOv10m | 0.765 | 0.556 |
| YOLOv11m | 0.813 | 0.604 |

## Published split (pretrained)

| Model | Precision | Recall | mAP50 | mAP50:95 | folds |
| --- | --- | --- | --- | --- | --- |
| YOLOv8m | 0.948 | 0.904 | 0.937 | 0.749 | 1 |
| YOLOv9m | 0.914 | 0.902 | 0.938 | 0.747 | 1 |
| YOLOv10m | 0.905 | 0.822 | 0.862 | 0.653 | 1 |
| YOLOv11m | 0.876 | 0.929 | 0.941 | 0.717 | 1 |

## Published split (from scratch)

| Model | Precision | Recall | mAP50 | mAP50:95 | folds |
| --- | --- | --- | --- | --- | --- |
| YOLOv8m | 0.933 | 0.852 | 0.951 | 0.748 | 1 |
| YOLOv9m | 0.918 | 0.917 | 0.930 | 0.742 | 1 |
| YOLOv10m | 0.780 | 0.947 | 0.934 | 0.734 | 1 |
| YOLOv11m | 0.888 | 0.909 | 0.952 | 0.764 | 1 |

## Effect of the evaluation protocol

| Model | published split mAP50 | grouped CV mAP50 | difference |
| --- | --- | --- | --- |
| YOLOv8m | 0.937 | 0.898 | -0.039 |
| YOLOv9m | 0.938 | 0.878 | -0.060 |
| YOLOv10m | 0.862 | 0.874 | +0.012 |
| YOLOv11m | 0.941 | 0.868 | -0.073 |

## Session-confound probe

Identical training data in both arms (intact session + cracked session eggs).
The test sets are cracked-only, so the broken-egg metrics are the measurement.
`cross` is episode 11, the only cracked egg recorded in the intact session;
`same` is episodes 18-19, cracked eggs from the cracked session.

| Model | cross recall | same recall | cross AP50 | same AP50 | AP50 difference |
| --- | --- | --- | --- | --- | --- |
| YOLOv8m | 0.778 | 0.833 | 0.877 | 0.956 | -0.079 |
| YOLOv9m | 1.000 | 0.909 | 0.995 | 0.944 | +0.051 |
| YOLOv10m | 0.995 | 0.901 | 0.984 | 0.933 | +0.051 |
| YOLOv11m | 0.997 | 0.889 | 0.995 | 0.972 | +0.023 |

Mean AP50 difference (cross - same): **+0.011** over 4 models. A session shortcut would make this strongly negative, because the cross arm's egg comes from a session in which training contained intact eggs only.

