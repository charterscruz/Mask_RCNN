# TRAIN
python boat.py --dataset /home/gcx/datasets/boat_voc_format/train --weights /home/gcx/repositories/mask-keras/Mask_RCNN/mask_rcnn_coco.h5 --logs /home/gcx/repositories/mask-keras/Mask_RCNN/ train

# DEPLOY
## IMAGE
python boat.py --dataset /home/gcx/datasets/boat_voc_format/train --weights /home/gcx/repositories/mask-keras/Mask_RCNN/boat20180423T2245/mask_rcnn_boat_0030.h5 --logs /home/gcx/repositories/mask-keras/Mask_RCNN/ --image /home/gcx/repositories/mask-keras/Mask_RCNN/images/source_img.jpg deploy

## VIDEO
--dataset /home/gcx/datasets/boat_voc_format/train --weights /home/gcx/repositories/mask-keras/Mask_RCNN/boat20180423T2245/mask_rcnn_boat_0030.h5 --logs /home/gcx/repositories/mask-keras/Mask_RCNN/ --video /home/gcx/gccruz_academiafa_edu_pt/workspace/matlab/datasets/videos_to_test/bigShipHighAlt_clip2.avi deploy