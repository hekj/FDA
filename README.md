<div align="center">

<h2>Frequency-enhanced Data Augmentation for Vision-and-Language Navigation</h2>

<div>
    <a href='https://scholar.google.com/citations?user=RHPI-NQAAAAJ&hl=zh-CN' target='_blank'>Keji He</a>;
    <a href='https://scholar.google.com/citations?hl=en&user=XdahAuoAAAAJ&view_op=list_works' target='_blank'>Chenyang Si</a>;
    <a href='https://zhihelu.github.io/'>Zhihe Lu</a>;
    <a href='https://yanrockhuang.github.io/' target='_blank'>Yan Huang</a>;
    <a href='http://scholar.google.com/citations?user=8kzzUboAAAAJ&hl=zh-CN' target='_blank'>Liang Wang</a>;
    <a href='https://sites.google.com/site/sitexinchaowang/?pli=1' target='_blank'>Xinchao Wang</a>
</div>


<h4 align="center">
  <a href="https://openreview.net/pdf?id=eKFrXWb0sT" target='_blank'>Paper</a>,
  <a href="https://openreview.net/attachment?id=eKFrXWb0sT&name=supplementary_material" target='_blank'>Supplementary Material</a>
</h4>

<h3><strong>Accepted to <a href='https://neurips.cc/' target='_blank'>NeurIPS 2023</a></strong></h3>


</div>

## Abstract

Vision-and-Language Navigation (VLN) is a challenging task that requires an agent to navigate through complex environments based on natural language instructions. In contrast to conventional approaches, which primarily focus on the spatial domain exploration, we propose a paradigm shift toward the Fourier domain. This alternative perspective aims to enhance visual-textual matching, ultimately improving the agent’s ability to understand and execute navigation tasks based on the given instructions. In this study, we first explore the significance of high-frequency information in VLN and provide evidence that it is instrumental in bolstering visual-textual matching processes. Building upon this insight, we further propose a sophisticated and versatile Frequency-enhanced Data Augmentation (FDA) technique to improve the VLN model’s capability of capturing critical high-frequency information. Specifically, this approach requires the agent to navigate in environments where only a subset of high-frequency visual information corresponds with the provided textual instructions, ultimately fostering the agent’s ability to selectively discern and capture pertinent high-frequency features according to the given instructions. Promising results on R2R, RxR, CVDN and REVERIE demonstrate that our FDA can be readily integrated with existing VLN approaches, improving performance without adding extra parameters, and keeping models simple and efficient.

## Method

<div  align="center">    
<img src="./method.jpg" width = "800" height = "370" alt="method" align=center />
</div>


## TODOs
* [X] Release Feature Extract Code.
* [X] Release Extracted Feature (by HAMT-ViT).
* [X] Release VLN Code (TD-STP on R2R).

## Setup

### Installation

This repo keeps the same installation settings as the [TD-STP](https://github.com/YushengZhao/TD-STP/tree/main) and [HAMT](https://github.com/cshizhe/VLN-HAMT?tab=readme-ov-file#extracting-features-optional). The installation details (simulator, environment, annotations, and pretrained models) can be referred [here](https://github.com/cshizhe/VLN-HAMT?tab=readme-ov-file#extracting-features-optional).

### Extracting features
The normal visual feature `vitbase_r2rfte2e_rgb.hdf5` and FDA visual feature `vitbase_r2rfte2e_fda.hdf5` can be downloaded [here](https://drive.google.com/drive/folders/1tRGJNJ53s9QoxSCqcl_7Ch3rtebf7RIQ?usp=sharing) directly. Then put them in the `/path_to_datasets/datasets/R2R/features/` directory.

If you want to extract them by yourself, please use the following commands with the scripts in `preprocess` directory.


   ```bash
   # Extract the normal visual feature
   CUDA_VISIBLE_DEVICES=0 python precompute_img_features_vit.py \
    --model_name vit_base_patch16_224 --out_image_logits \
    --connectivity_dir /path_to_datasets/datasets/R2R/connectivity \
    --scan_dir /path_to_datasets/datasets/Matterport3D/v1_unzip_scans \
    --num_workers 4 \
    --checkpoint_file /path_to_datasets/datasets/R2R/trained_models/vit_step_22000.pt \
    --output_file /path_to_datasets/datasets/R2R/features/vitbase_r2rfte2e_rgb.hdf5

   # Extract the FDA visual feature
   CUDA_VISIBLE_DEVICES=0 python fda_highfreq_perturbed.py \
   --model_name vit_base_patch16_224 --out_image_logits \
   --connectivity_dir /path_to_datasets/R2R/connectivity \
   --scan_dir /path_to_datasets/datasets/Matterport3D/v1_unzip_scans \
   --num_workers 4 \
   --checkpoint_file /path_to_datasets/datasets/R2R/trained_models/vit_step_22000.pt \
   --output_file /path_to_datasets/datasets/R2R/features/vitbase_r2rfte2e_fda.hdf5
   ```

## Training & Inference

To train on R2R: 
```
cd finetune_src
bash ./scripts/run_r2r.sh
```

To test on R2R:
```
cd finetune_src
bash ./scripts/test_r2r.sh
```



# Acknowledge

Our implementations are partially inspired by [TD-STP](https://github.com/YushengZhao/TD-STP/tree/main) and [HAMT](https://github.com/cshizhe/VLN-HAMT?tab=readme-ov-file#extracting-features-optional). We appreciate them for releasing their great works!

# Citation

If you find this repository useful, please consider citing our paper:

```
@inproceedings{
he2023frequencyenhanced,
title={Frequency-Enhanced Data Augmentation for Vision-and-Language Navigation},
author={Keji He and Chenyang Si and Zhihe Lu and Yan Huang and Liang Wang and Xinchao Wang},
booktitle={Thirty-seventh Conference on Neural Information Processing Systems},
year={2023}
}
```

