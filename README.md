# Selective Synaptic Dampening (AAAI + ICLR TP code)

![GitHub last commit (branch)](https://img.shields.io/github/last-commit/if-loops/selective-synaptic-dampening/main) ![GitHub Repo stars](https://img.shields.io/github/stars/if-loops/selective-synaptic-dampening) ![GitHub repo size](https://img.shields.io/github/repo-size/if-loops/selective-synaptic-dampening)




![SSD_heading](https://github.com/if-loops/selective-synaptic-dampening/assets/47212405/2abb0ef1-8646-479e-a00e-613960d27f9c)

This is the code for the paper **Fast Machine Unlearning Without Retraining Through Selective Synaptic Dampening** (https://arxiv.org/abs/2308.07707), accepted at The 38th Annual **AAAI Conference on Artificial Intelligence** (Main Track).

## Related research

| Paper  | Code | Venue/Status |
| ------------- | ------------- |  ------------- |
| [Potion: Towards Poison Unlearning](https://arxiv.org/abs/2406.09173) | [GitHub](https://github.com/if-loops/towards_poison_unlearning) |  Journal of Data-Centric Machine Learning Research (DMLR)  |
| [Zero-Shot Machine Unlearning at Scale via Lipschitz Regularization](https://browse.arxiv.org/abs/2402.01401)  | [GitHub](https://github.com/jwf40/Zeroshot-Unlearning-At-Scale) |  Preprint  |
| [Parameter-Tuning-Free Data Entry Error Unlearning with Adaptive Selective Synaptic Dampening](https://arxiv.org/abs/2402.10098)  | [GitHub](https://github.com/if-loops/adaptive-selective-synaptic-dampening) |  Preprint  |
| [ Loss-Free Machine Unlearning](https://arxiv.org/abs/2402.19308) (i.e. Label-Free) -> LFSSD | see below |  ICLR 2024 Tiny Paper  |

### Implementing LFSSD:
Replace the following in the compute_importances function(s):

```
# Vanilla SSD:
criterion = nn.CrossEntropyLoss()
loss = criterion(out, y)
...
imp.data += p.grad.data.clone().pow(2)

# LFSSD:
loss = torch.norm(out, p="fro", dim=1).pow(2).mean()
...
imp.data += p.grad.data.clone().abs()
```

## Usage

리팩토링 후에는 저장소 루트에서 `src/run_experiments.sh` 하나로 실험을 실행합니다.

```
./src/run_experiments.sh cifar10-random-resnet 0 42
```

인자는 순서대로 `실험 이름`, `GPU 번호`, `seed`입니다. 위 예시는 GPU 0에서
`cifar10-random-resnet` 실험을 seed 42로 실행합니다.

실행 가능한 실험 이름은 아래 명령어로 확인할 수 있습니다.

```
./src/run_experiments.sh --help
```

현재 지원하는 실험 이름은 다음과 같습니다.

```
cifar10-random-resnet
cifar10-random-vit
cifar20-fullclass-resnet
cifar20-fullclass-vit
cifar20-subclass-resnet
cifar20-subclass-vit
cifar100-fullclass-resnet
cifar100-fullclass-vit
pins-fullclass-resnet
all-resnet
all-vit
all
```

각 실험을 실행하기 전에 필요한 pretrained weight 경로가 설정되어 있어야 합니다.
기본 경로가 없는 실험은 환경변수로 weight 경로를 넘겨 실행합니다.

```
CIFAR20_RESNET_WEIGHT_PATH=/path/to/model.pth ./src/run_experiments.sh cifar20-fullclass-resnet 0 42
```

사용 가능한 weight path 환경변수는 다음과 같습니다.

```
CIFAR10_RESNET_WEIGHT_PATH
CIFAR10_VIT_WEIGHT_PATH
CIFAR20_RESNET_WEIGHT_PATH
CIFAR20_VIT_WEIGHT_PATH
CIFAR100_RESNET_WEIGHT_PATH
CIFAR100_VIT_WEIGHT_PATH
PINS_RESNET_WEIGHT_PATH
```

실행 커맨드 예시는 다음과 같습니다.

```bash
# CIFAR10 random forgetting, ResNet18, GPU 0, seed 42
./src/run_experiments.sh cifar10-random-resnet 0 42

# CIFAR10 random forgetting, ViT
CIFAR10_VIT_WEIGHT_PATH=/path/to/vit_cifar10.pth ./src/run_experiments.sh cifar10-random-vit 0 42

# CIFAR20 full-class forgetting, ResNet18
CIFAR20_RESNET_WEIGHT_PATH=/path/to/resnet_cifar20.pth ./src/run_experiments.sh cifar20-fullclass-resnet 0 42

# CIFAR20 subclass forgetting, ResNet18
CIFAR20_RESNET_WEIGHT_PATH=/path/to/resnet_cifar20.pth ./src/run_experiments.sh cifar20-subclass-resnet 0 42

# CIFAR100 full-class forgetting, ViT
CIFAR100_VIT_WEIGHT_PATH=/path/to/vit_cifar100.pth ./src/run_experiments.sh cifar100-fullclass-vit 0 42

# Pins Face Recognition full-class forgetting, ResNet18
PINS_RESNET_WEIGHT_PATH=/path/to/pins_resnet.pth ./src/run_experiments.sh pins-fullclass-resnet 0 42

# ResNet 계열 실험 전체 실행
CIFAR10_RESNET_WEIGHT_PATH=/path/to/resnet_cifar10.pth \
CIFAR20_RESNET_WEIGHT_PATH=/path/to/resnet_cifar20.pth \
CIFAR100_RESNET_WEIGHT_PATH=/path/to/resnet_cifar100.pth \
PINS_RESNET_WEIGHT_PATH=/path/to/pins_resnet.pth \
./src/run_experiments.sh all-resnet 0 42

# ViT 계열 실험 전체 실행
CIFAR10_VIT_WEIGHT_PATH=/path/to/vit_cifar10.pth \
CIFAR20_VIT_WEIGHT_PATH=/path/to/vit_cifar20.pth \
CIFAR100_VIT_WEIGHT_PATH=/path/to/vit_cifar100.pth \
./src/run_experiments.sh all-vit 0 42
```

스크립트는 각 Python 명령어를 실행하기 전에 터미널에 출력하고, 실험이 끝나면
`TestAcc`, `RetainTestAcc`, `ZRF`, `MIA`, `Df`, `MethodTime` 값을 이름과 함께
출력합니다.

기존 개별 실험 스크립트도 `src/` 안에 그대로 남아 있습니다. 필요하면 예전처럼
`cd src` 후 `./cifar10_random_exps.sh 0 42` 형태로 직접 실행할 수 있습니다.
Windows/Unix 줄바꿈 차이로 실행 문제가 생기면 `dos2unix "filename"`을 사용하세요.

## Setup

You will need to train ResNet18's and Vision Transformers. Use pretrain_model.py for this and then copy the paths of the models into the respecive .sh files.

```
# fill in _ with your desired parameters as described in pretrain_model.py
python pretrain_model.py -net _ -dataset _ -classes _ -gpu _
```

We used https://hub.docker.com/layers/tensorflow/tensorflow/latest-gpu-py3-jupyter/images/sha256-901b827b19d14aa0dd79ebbd45f410ee9dbfa209f6a4db71041b5b8ae144fea5 as our base image and installed relevant packages on top.

```
datetime
wandb
sklearn
torch
copy
tqdm
transformers
matplotlib
scipy
```

You will need a wandb.ai account to use the implemented logging. Feel free to replace with any other logger of your choice.

## Modifying SSD

SSD functions are in ssd.py. To change alpha and lambda, set them in the respective forget_..._main.py file per unlearning task.

## Citing this work

```
@article{Foster_Schoepf_Brintrup_2024,
      title={Fast Machine Unlearning without Retraining through Selective Synaptic Dampening},
      volume={38},
      url={https://ojs.aaai.org/index.php/AAAI/article/view/29092},
      DOI={10.1609/aaai.v38i11.29092},
      number={11},
      journal={Proceedings of the AAAI Conference on Artificial Intelligence},
      author={Foster, Jack and Schoepf, Stefan and Brintrup, Alexandra},
      year={2024},
      month={Mar.},
      pages={12043-12051} }
```

## Authors

For our newest research, feel free to follow our socials:

Jack Foster: [LinkedIn](https://www.linkedin.com/in/jackfoster-ml/), [Twitter](https://twitter.com/JackFosterML)  

Stefan Schoepf: [LinkedIn](https://www.linkedin.com/in/schoepfstefan/), [Twitter](https://twitter.com/S__Schoepf)  

Alexandra Brintrup: [LinkedIn](https://www.linkedin.com/in/alexandra-brintrup-1684171/)  

Supply Chain AI Lab: [LinkedIn](https://www.linkedin.com/company/supply-chain-ai-lab/)  
