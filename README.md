This is the Course Assignment Project of GNR:638 Deep Learning for Computer Vision.  
The task is to Deblur the images which has been blurred by Gaussian Kernel of varying sizes.  
We first tried to deblur with the help of Famous U-Net Architecture of small size having 2.6 million parameters , It shows a PSNR score of 9.8 which is not so great.  
We tried Transformer based architecture namely Stripformer , the code for the same is in the Folder Stripformer. This Stripformer was modified for 14.9 M Parameters.  
This Stripformer worked better than CNN based architecture.

@inproceedings{Tsai2022Stripformer,
  author    = {Fu-Jen Tsai and Yan-Tsung Peng and Yen-Yu Lin and Chung-Chi Tsai and Chia-Wen Lin},
  title     = {Stripformer: Strip Transformer for Fast Image Deblurring},
  booktitle = {ECCV},
  year      = {2022}
}
