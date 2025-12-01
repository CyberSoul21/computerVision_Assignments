import numpy as np

npz = np.load('out/image1_image2_matches.npz')  # or put the exact filename
mask = npz['matches'] > -1
x1 = npz['keypoints0'][mask]
x2 = npz['keypoints1'][npz['matches'][mask]]
conf = npz['match_confidence'][mask]
print(len(x1), "matches")



"""
cd SuperGluePretrainedNetwork

python match_pairs.py \
  --resize 752 \
  --superglue indoor \
  --max_keypoints 2048 \
  --nms_radius 3 \
  --resize_float \
  --input_dir .. \
  --input_pairs ../pairs/euroc_sample_pairs.txt \
  --output_dir ../out \
  --viz
  --force_cpu
"""