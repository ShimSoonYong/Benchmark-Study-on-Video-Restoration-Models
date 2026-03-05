# general settings
name: TRAIN_OPS
data_dim: 2
resume: -1 #  -1: start from scratch, 0: resume from the latest checkpoint, 1: resume from best checkpoint
# data settings
datasets:
  train:
    name: KernelWizard_train
    type: GOPRO_Large
    gt_dir: /workspace/GOPRO_Large/train/*/sharp
    input_dir: /workspace/GOPRO_Large/train/*/blur
    repeat: 160
    crop_size: 256
    augment: true

    # data loader settings
    batch_size: 4
    num_workers: 4

  val:
    gt_dir: /workspace/GOPRO_Large/test/*/sharp
    input_dir: /workspace/GOPRO_Large/test/*/blur
    repeat: 1
    batch_size: 1
    num_workers: 4
    crop_size: 256
  
  test:
    gt_dir: /workspace/GOPRO_Large/test/*/sharp
    input_dir: /workspace/GOPRO_Large/test/*/blur
    repeat: 1
    batch_size: 1
    num_workers: 4
    crop_size: 256
    
model:
  type: KernelWizard
  input_nc: 3
  nf: 64
  front_RBs: 10
  back_RBs: 20
  N_frames: 1
  kernel_dim: 512
  img_size: 256
  use_vae: false
  KernelExtractor:
    norm: none
    use_sharp: true
    n_blocks: 4
    padding_type: reflect
    use_dropout: false
  Adapter:
    norm: none
    use_dropout: false

# training settings
train:
  details:
    epochs: 1000
    warmup_iter: -1 # no warm up
    use_grad_clip: true
    grad_norm: !!float 0.01
    AMP: true

  # Split 300k iterations into two cycles. 
  # 1st cycle: fixed 3e-4 LR for 92k iters. 
  # 2nd cycle: cosine annealing (3e-4 to 1e-6) for 208k iters.
  scheduler:
    type: MultiStepLR
    milestones: [120, 180, 240, 270]
    gamma: 0.5



  optimizer:
    type: AdamW
    lr: !!float 1e-4
    weight_decay: 0
    betas: [0.9, 0.999]   
  

  # losses
  pixel_opt:
    type: L1Loss
    # loss앞에 곱해지는 weight값
    loss_weight: 1
    # output의 sum을 할지, mean을 할지, none을 할지(이러면 아마도 batch 개수만큼의 값을 가진 vector로 return되는 듯?)
    reduction: mean

# test settings
test:
  save_img: false
  test_every: 1
  metrics: [RMSE, PSNR]
  crop: true
  tile: [1, 256, 256]
  tile_overlap: [0, 10, 0]

  