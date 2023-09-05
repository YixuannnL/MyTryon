# MyTryon

TryOnDiffusion/
|
|-- Datasets/                          # Directory to store datasets
|   |-- scripts/                   # Scripts for processing datasets
|   |-- raw/                       # Raw, unprocessed data
|   |-- processed/                 # Preprocessed data and datasets ready for training/testing 
|   |   |-- images/                    # Images to add noise
|   |   |-- agnostic/                  # Agnostic images
|   |   |-- segmented/                 # Segmented garment
|   |   |-- pose_g/                    # Garment pose
|   |   |-- pose_p/                    # Person pose
|
|-- models/                        # Directory for model architectures
|   |-- Unet.py                    # Parallel-UNet/Efficient-Unet model definitions
|   |-- unet_attention.py          # Transformer/attention related definitions
|
|-- checkpoints/                   # Directory to save model checkpoints during training
|
|-- logs/                          # Directory for logging, tensorboard logs, etc.
|
|-- train.py                       # Main training and evaluation script
|-- opt.py                         # Parsing Command Line Parameters
|-- requirements.txt               # List of dependencies and required libraries
|-- README.md                      # Project documentation and instructions
