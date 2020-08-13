# AdaptNet
An adapt transform method for CT HU values. It could search the most approriate window level and window width for the current task.
###### Illustration
![image](https://github.com/LWHYC/AdaptNet/blob/master/Adapt_trans.png)

###### Example Code
```
from Adapt_transform import Adapt_transform2

hu_lis  = np.asarray([[-2, 4]]) # the initial window level
norm_lis  = np.asarray([[1, 1]]) 
smooth_lis = np.asarray([[4,4]])
adapt_trans = Adapt_transform2(hu_lis=hu_lis, norm_lis = norm_lis, smooth_lis=smooth_lis)
adapt_trans.train()
transformed_img_batch = adapt_trans(img_batch) # img_batch is your untransformed ct images
```
