"""Class-to-prompt mappings for ImageNet-100 and CIFAR-100 datasets."""

# ImageNet-100 WordNet IDs
IMAGENET100_CLASS_IDS = [
    'n01498041', 'n01514859', 'n01582220', 'n01608432', 'n01616318',
    'n01687978', 'n01776313', 'n01806567', 'n01833805', 'n01882714',
    'n01910747', 'n01944390', 'n01985128', 'n02007558', 'n02071294',
    'n02085620', 'n02114855', 'n02123045', 'n02128385', 'n02129165',
    'n02129604', 'n02165456', 'n02190166', 'n02219486', 'n02226429',
    'n02279972', 'n02317335', 'n02326432', 'n02342885', 'n02363005',
    'n02391049', 'n02395406', 'n02403003', 'n02422699', 'n02442845',
    'n02444819', 'n02480855', 'n02510455', 'n02640242', 'n02672831',
    'n02687172', 'n02701002', 'n02730930', 'n02769748', 'n02782093',
    'n02787622', 'n02793495', 'n02799071', 'n02802426', 'n02814860',
    'n02840245', 'n02906734', 'n02948072', 'n02980441', 'n02999410',
    'n03014705', 'n03028079', 'n03032252', 'n03125729', 'n03160309',
    'n03179701', 'n03220513', 'n03249569', 'n03291819', 'n03384352',
    'n03388043', 'n03450230', 'n03481172', 'n03594734', 'n03594945',
    'n03627232', 'n03642806', 'n03649909', 'n03661043', 'n03676483',
    'n03724870', 'n03733281', 'n03759954', 'n03761084', 'n03773504',
    'n03804744', 'n03916031', 'n03938244', 'n04004767', 'n04026417',
    'n04090263', 'n04133789', 'n04153751', 'n04296562', 'n04330267',
    'n04371774', 'n04404412', 'n04465501', 'n04485082', 'n04507155',
    'n04536866', 'n04579432', 'n04606251', 'n07714990', 'n07745940',
]

# ImageNet-100 human-readable class names (aligned with IMAGENET100_CLASS_IDS)
IMAGENET100_CLASS_NAMES = [
    'stingray', 'hen', 'magpie', 'kite', 'vulture',
    'agama', 'tick', 'quail', 'hummingbird', 'koala',
    'jellyfish', 'snail', 'crayfish', 'flamingo', 'killer whale',
    'Chihuahua', 'coyote', 'tabby cat', 'leopard', 'lion',
    'tiger', 'ladybug', 'fly', 'ant', 'grasshopper',
    'monarch butterfly', 'starfish', 'hare', 'hamster', 'beaver',
    'zebra', 'hog', 'ox', 'impala', 'mink',
    'otter', 'gorilla', 'giant panda', 'sturgeon', 'accordion',
    'aircraft carrier', 'ambulance', 'apron', 'backpack', 'balloon',
    'banjo', 'barn', 'baseball', 'basketball', 'beacon',
    'binder', 'broom', 'candle', 'castle', 'chain',
    'chest', 'church', 'cinema', 'cradle', 'dam',
    'desk', 'dome', 'drum', 'envelope', 'forklift',
    'fountain', 'gown', 'hammer', 'jean', 'jeep',
    'knot', 'laptop', 'lawn mower', 'library', 'lipstick',
    'mask', 'maze', 'microphone', 'microwave', 'missile',
    'nail', 'perfume', 'pillow', 'printer', 'purse',
    'rifle', 'sandal', 'screw', 'stage', 'stove',
    'swing', 'television', 'tractor', 'tripod', 'umbrella',
    'violin', 'whistle', 'wreck', 'broccoli', 'strawberry',
]

# CIFAR-100 class IDs (string indices 0-99)
CIFAR100_CLASS_IDS = [str(i) for i in range(100)]

# CIFAR-100 human-readable class names
CIFAR100_CLASS_NAMES = [
    'apples', 'aquarium fish', 'baby', 'bear', 'beaver',
    'bed', 'bee', 'beetle', 'bicycle', 'bottles',
    'bowls', 'boy', 'bridge', 'bus', 'butterfly',
    'camel', 'cans', 'castle', 'caterpillar', 'cattle',
    'chair', 'chimpanzee', 'clock', 'cloud', 'cockroach',
    'couch', 'crab', 'crocodile', 'cups', 'dinosaur',
    'dolphin', 'elephant', 'flatfish', 'forest', 'fox',
    'girl', 'hamster', 'house', 'kangaroo', 'computer keyboard',
    'lamp', 'lawn-mower', 'leopard', 'lion', 'lizard',
    'lobster', 'man', 'maple', 'motorcycle', 'mountain',
    'mouse', 'mushrooms', 'oak', 'oranges', 'orchids',
    'otter', 'palm', 'pears', 'pickup truck', 'pine',
    'plain', 'plates', 'poppies', 'porcupine', 'possum',
    'rabbit', 'raccoon', 'ray', 'road', 'rocket',
    'roses', 'sea', 'seal', 'shark', 'shrew',
    'skunk', 'skyscraper', 'snail', 'snake', 'spider',
    'squirrel', 'streetcar', 'sunflowers', 'sweet peppers', 'table',
    'tank', 'telephone', 'television', 'tiger', 'tractor',
    'train', 'trout', 'tulips', 'turtle', 'wardrobe',
    'whale', 'willow', 'wolf', 'woman', 'worm',
]

# Lookup dictionaries: class_id -> class_name
IMAGENET100_ID_TO_NAME = dict(zip(IMAGENET100_CLASS_IDS, IMAGENET100_CLASS_NAMES))
CIFAR100_ID_TO_NAME = dict(zip(CIFAR100_CLASS_IDS, CIFAR100_CLASS_NAMES))


def get_target_class_list(dataset: str) -> list:
    """Return the ordered list of class IDs for a given dataset.

    Args:
        dataset: One of 'imagenet100' or 'cifar100'.

    Returns:
        List of class ID strings.
    """
    if dataset == 'imagenet100':
        return IMAGENET100_CLASS_IDS
    elif dataset == 'cifar100':
        return CIFAR100_CLASS_IDS
    else:
        raise ValueError(f"Unknown dataset: {dataset}. Supported: 'imagenet100', 'cifar100'.")


def get_prompt(dataset: str, target_class: str) -> str:
    """Build a text prompt for the diffusion model given a dataset and class.

    Args:
        dataset: One of 'imagenet100' or 'cifar100'.
        target_class: The class ID (e.g., 'n01498041' for ImageNet-100, '0' for CIFAR-100).

    Returns:
        A descriptive text prompt string.
    """
    if dataset == 'imagenet100':
        class_name = IMAGENET100_ID_TO_NAME.get(target_class)
        if class_name is None:
            raise ValueError(f"Unknown ImageNet-100 class ID: {target_class}")
        return f"A high-quality image of the {class_name}"
    elif dataset == 'cifar100':
        class_name = CIFAR100_ID_TO_NAME.get(target_class)
        if class_name is None:
            raise ValueError(f"Unknown CIFAR-100 class ID: {target_class}")
        return f"A high-quality image of the {class_name}"
    else:
        # Fallback generic prompt for unknown datasets
        return (
            "A highly detailed, high-resolution masterpiece with intricate textures "
            "and sharp details. Ultra-clear rendering, no artifacts, perfect clarity, "
            "smooth edges, and well-balanced composition."
        )
