#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

usage() {
    cat <<'USAGE'
Usage:
  ./run_experiments.sh [experiment] [gpu] [seed]

Arguments:
  experiment  Experiment name. Default: cifar10-random-resnet
  gpu         GPU id passed to CUDA_VISIBLE_DEVICES. Default: 0
  seed        Random seed. Default: 42

Experiments:
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

Weight path overrides:
  CIFAR10_RESNET_WEIGHT_PATH
  CIFAR10_VIT_WEIGHT_PATH
  CIFAR20_RESNET_WEIGHT_PATH
  CIFAR20_VIT_WEIGHT_PATH
  CIFAR100_RESNET_WEIGHT_PATH
  CIFAR100_VIT_WEIGHT_PATH
  PINS_RESNET_WEIGHT_PATH
USAGE
}

EXPERIMENT="${1:-cifar10-random-resnet}"
DEVICE="${2:-0}"
SEED="${3:-42}"
PAUSE_SECONDS="${PAUSE_SECONDS:-10}"

if [[ "$EXPERIMENT" == "-h" || "$EXPERIMENT" == "--help" ]]; then
    usage
    exit 0
fi

run_cmd() {
    echo
    echo "================================================================================"
    echo "Running: $*"
    echo "================================================================================"
    CUDA_VISIBLE_DEVICES="$DEVICE" python -u "$@"
    sleep "$PAUSE_SECONDS"
}

require_weight_path() {
    local weight_path="$1"
    local label="$2"

    if [[ -z "$weight_path" ]]; then
        echo "Missing weight path for $label." >&2
        echo "Set the matching *_WEIGHT_PATH environment variable before running." >&2
        exit 1
    fi

    if [[ ! -f "$weight_path" ]]; then
        echo "Weight path for $label does not exist: $weight_path" >&2
        exit 1
    fi
}

run_random_forgetting() {
    local model="$1"
    local dataset="Cifar10"
    local classes="10"
    local forget_perc="0.00165"
    local weight_path="$2"
    local methods=(baseline ssd_tuning finetune amnesiac blindspot retrain)

    require_weight_path "$weight_path" "$dataset $model"

    for method in "${methods[@]}"; do
        run_cmd forget_random_main.py \
            -net "$model" \
            -dataset "$dataset" \
            -classes "$classes" \
            -gpu \
            -method "$method" \
            -forget_perc "$forget_perc" \
            -weight_path "$weight_path" \
            -seed "$SEED"
    done
}

run_fullclass_forgetting() {
    local model="$1"
    local dataset="$2"
    local classes="$3"
    local weight_path="$4"
    shift 4
    local forget_classes=("$@")
    local methods=(baseline ssd_tuning finetune amnesiac blindspot UNSIR retrain)

    if [[ "$model" == "ViT" ]]; then
        methods=(blindspot baseline ssd_tuning finetune amnesiac UNSIR retrain)
    fi

    require_weight_path "$weight_path" "$dataset $model"

    for forget_class in "${forget_classes[@]}"; do
        for method in "${methods[@]}"; do
            run_cmd forget_full_class_main.py \
                -net "$model" \
                -dataset "$dataset" \
                -classes "$classes" \
                -gpu \
                -method "$method" \
                -forget_class "$forget_class" \
                -weight_path "$weight_path" \
                -seed "$SEED"
        done
    done
}

run_subclass_forgetting() {
    local model="$1"
    local weight_path="$2"
    shift 2
    local forget_classes=("$@")
    local methods=(baseline ssd_tuning finetune amnesiac blindspot UNSIR retrain)

    if [[ "$model" == "ViT" ]]; then
        methods=(blindspot baseline ssd_tuning finetune amnesiac UNSIR retrain)
    fi

    require_weight_path "$weight_path" "Cifar20 subclass $model"

    for forget_class in "${forget_classes[@]}"; do
        for method in "${methods[@]}"; do
            run_cmd forget_subclass_main.py \
                -net "$model" \
                -dataset Cifar20 \
                -superclasses 20 \
                -subclasses 100 \
                -gpu \
                -method "$method" \
                -forget_class "$forget_class" \
                -weight_path "$weight_path" \
                -seed "$SEED"
        done
    done
}

run_experiment() {
    case "$1" in
        cifar10-random-resnet)
            run_random_forgetting ResNet18 "${CIFAR10_RESNET_WEIGHT_PATH:-./checkpoint/ResNet18/Thursday_12_March_2026_07h_22m_55s/ResNet18-Cifar10-18-best.pth}"
            ;;
        cifar10-random-vit)
            run_random_forgetting ViT "${CIFAR10_VIT_WEIGHT_PATH:-}"
            ;;
        cifar20-fullclass-resnet)
            run_fullclass_forgetting ResNet18 Cifar20 20 "${CIFAR20_RESNET_WEIGHT_PATH:-}" vehicle2 veg people electrical_devices natural_scenes
            ;;
        cifar20-fullclass-vit)
            run_fullclass_forgetting ViT Cifar20 20 "${CIFAR20_VIT_WEIGHT_PATH:-}" vehicle2 veg people electrical_devices natural_scenes
            ;;
        cifar20-subclass-resnet)
            run_subclass_forgetting ResNet18 "${CIFAR20_RESNET_WEIGHT_PATH:-}" rocket mushroom baby lamp sea
            ;;
        cifar20-subclass-vit)
            run_subclass_forgetting ViT "${CIFAR20_VIT_WEIGHT_PATH:-}" sea rocket mushroom baby lamp
            ;;
        cifar100-fullclass-resnet)
            run_fullclass_forgetting ResNet18 Cifar100 100 "${CIFAR100_RESNET_WEIGHT_PATH:-}" rocket mushroom baby lamp sea
            ;;
        cifar100-fullclass-vit)
            run_fullclass_forgetting ViT Cifar100 100 "${CIFAR100_VIT_WEIGHT_PATH:-}" rocket mushroom baby lamp sea
            ;;
        pins-fullclass-resnet)
            run_fullclass_forgetting ResNet18 PinsFaceRecognition 105 "${PINS_RESNET_WEIGHT_PATH:-checkpoint/ResNet18/Saturday_12_August_2023_10h_50m_04s/ResNet18-PinsFaceRecognition-194-best.pth}" 1 10 20 30 40
            ;;
        all-resnet)
            run_experiment cifar10-random-resnet
            run_experiment cifar20-subclass-resnet
            run_experiment cifar20-fullclass-resnet
            run_experiment cifar100-fullclass-resnet
            run_experiment pins-fullclass-resnet
            ;;
        all-vit)
            run_experiment cifar10-random-vit
            run_experiment cifar100-fullclass-vit
            run_experiment cifar20-fullclass-vit
            run_experiment cifar20-subclass-vit
            ;;
        all)
            run_experiment all-resnet
            run_experiment all-vit
            ;;
        *)
            echo "Unknown experiment: $1" >&2
            usage
            exit 1
            ;;
    esac
}

echo "Experiment: $EXPERIMENT"
echo "GPU: $DEVICE"
echo "Seed: $SEED"
run_experiment "$EXPERIMENT"
