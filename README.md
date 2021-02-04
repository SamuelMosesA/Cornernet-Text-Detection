# Cornernet-Text-Detection

## Citation
**CornerNet: Detecting Objects as Paired Keypoints**  
Hei Law, Jia Deng  
European Conference on Computer Vision (ECCV), 2018  
Code based on https://github.com/princeton-vl/CornerNet

## Dataloader Details
The data loader must return these as target
- target heatmaps for the top-left(tl) and bottom-right(br) output with a gaussian surface at each keypoint
- tl and br tag indices - the indices of the keypoints after flattening it ie. x,y will have index y*width + x  
- tl and br offsets - because we are reducing the dimension of the image 4 times
- tag_masks -  we will be using a fixed tensor for the tags. since each pic will have
different number of objects the number will differ. we will denote the number of tags
  by an torch array l with l[0] to l[#tags -1] as 1 and the rest as zero
  
all these must be in a list of tensors in the same order given above

## Model Details
- the cornernet model takes 2 inputs while training 
  - the image(it can be large as it is made smaller 4 times inside)
  - the targets(for the tl br indices)
  
- and the image alone as input while in eval mode


you can get the idea for how the target is generated in the 
/sample/coco.py in the github repo in the citation
