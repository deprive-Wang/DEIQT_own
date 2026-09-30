"""与作者代码对齐的图像预处理；输入由数据集转换为 RGB 的 PIL 图像。"""

from torchvision import transforms


def build_transform(dataset: str, is_train: bool) -> transforms.Compose:
    """将图像转换为标准化后的 [3, 224, 224] Tensor。

    训练使用随机水平翻转；训练和测试均使用随机裁剪。
    每次调用只生成一个裁剪，多裁剪采样及分数汇总由数据与评估模块负责。
    """
    supported = {"koniq", "livec", "live", "tid2013", "csiq", "kadid", "spaq", "livefb"}
    if dataset not in supported:
        raise ValueError(f"不支持的数据集 {dataset!r}；可选：{sorted(supported)}")

    operations = []
    if is_train:
        operations.append(transforms.RandomHorizontalFlip(p=0.5))
    if dataset == "koniq":
        # 保留作者的顺序和尺寸：先翻转再缩放，(512, 384) 表示高、宽。
        operations.append(transforms.Resize((512, 384)))

    operations.extend([
        # 测试也随机裁剪；不改成 CenterCrop，不对过小图像静默补边或拉伸。
        transforms.RandomCrop((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225),
        ),
    ])
    return transforms.Compose(operations)
