CUDA_VISIBLE_DEVICES=0  python main.py -d 'dataset' -dset imagenet -a resdg18 --checkpoint model_ckpt \
     --pretrained ./model_ckpt/mindspore_PACS_art_painting_model_best.ckpt --source 'art_painting' --single --evaluate
CUDA_VISIBLE_DEVICES=0  python main.py -d 'dataset' -dset imagenet -a resdg18 --checkpoint model_ckpt \
     --pretrained ./model_ckpt/mindspore_PACS_cartoon_model_best.ckpt --source 'cartoon' --single --evaluate
CUDA_VISIBLE_DEVICES=0  python main.py -d 'dataset' -dset imagenet -a resdg18 --checkpoint model_ckpt \
     --pretrained ./model_ckpt/mindspore_PACS_photo_model_best.ckpt --source 'photo' --single --evaluate
CUDA_VISIBLE_DEVICES=0  python main.py -d 'dataset' -dset imagenet -a resdg18 --checkpoint model_ckpt \
     --pretrained ./model_ckpt/mindspore_PACS_sketch_model_best.ckpt --source 'sketch' --single --evaluate
     