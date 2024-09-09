## MSc individual project by Yilun Cheng

## Acknowledgement
This project has intensively borrow codes from the following repositories. Many thanks to the authors for sharing their codes.
- [gaussian splatting](https://github.com/graphdeco-inria/gaussian-splatting)


## Data structure accepted:

- <data_folder>
- |---|---dataset_name
- |---|---|---diff
- |---|---|---|---images
- |---|---|---|---sparse
- |---|---|---|---|---0
- |---|---|---spec
- |---|---|---|---images
- |---|---|---|---sparse
- |---|---|---|---|---0
- |---|---|---all
- |---|---|---|---images
- |---|---|---|---sparse
- |---|---|---|---|---0

where `images` contains the input images of corresponding shading components, and `0` contains the sparse model generated from COLMAP. 