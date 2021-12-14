# NarrowDNN

Thanks to non-convexity, even very narrow DNN (with only 1 or 2 channels) could perform well in some abstract binary classification tasks.

> So what if we aggregate a lot of 1(or 2)-channel DNN chains to handle multi-classification tasks (e.g. MNIST, Fashion-MNIST, CIFAR-10)? Let's see.

![](assets/workflow.png)

## Results

### Overall Accuracy

| Dataset           | Accuracy    | Arch      | #Param |
| ----------------- | ----------- | --------- | ------ |
| **MNIST**         | 93.40%      | 1-channel | 130    |
| **Fashion-MNIST** | 80.39%      | 1-channel | 130    |
| **CIFAR-10**      | 47.72%      | 2-channel | 493    |

### Per-class Accuracy

![](assets/per-class-acc-mnist.png)

![](assets/per-class-acc-fashionmnist.png)

![](assets/per-class-acc-cifar10.png)

## Details

* Only convolution layers are adopted.
* The *1-channel* DNN chain (for MNIST and Fashion-MNIST) architecture:
    ```python
    narrow_VGG(
        (features): Sequential(
            (0): Conv2d(1, 1, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            (1): ReLU(inplace=True)
            (2): Conv2d(1, 1, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            (3): ReLU(inplace=True)
            (4): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
            (5): Conv2d(1, 1, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            (6): ReLU(inplace=True)
            (7): Conv2d(1, 1, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            (8): ReLU(inplace=True)
            (9): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
            (10): Conv2d(1, 1, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            (11): ReLU(inplace=True)
            (12): Conv2d(1, 1, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            (13): ReLU(inplace=True)
            (14): Conv2d(1, 1, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            (15): ReLU(inplace=True)
            (16): MaxPool2d(kernel_size=2, stride=1, padding=0, dilation=1, ceil_mode=False)
            (17): Conv2d(1, 1, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            (18): ReLU(inplace=True)
            (19): Conv2d(1, 1, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            (20): ReLU(inplace=True)
            (21): Conv2d(1, 1, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            (22): ReLU(inplace=True)
            (23): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
            (24): Conv2d(1, 1, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            (25): ReLU(inplace=True)
            (26): Conv2d(1, 1, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            (27): ReLU(inplace=True)
            (28): Conv2d(1, 1, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            (29): ReLU(inplace=True)
            (30): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
        )
    )
    ```
* The *2-channel* DNN chain (for CIFAR-10) architecture:
    ```python
    narrow_VGG(
        (features): Sequential(
            (0): Conv2d(3, 2, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            (1): ReLU(inplace=True)
            (2): Conv2d(2, 2, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            (3): ReLU(inplace=True)
            (4): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
            (5): Conv2d(2, 2, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            (6): ReLU(inplace=True)
            (7): Conv2d(2, 2, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            (8): ReLU(inplace=True)
            (9): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
            (10): Conv2d(2, 2, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            (11): ReLU(inplace=True)
            (12): Conv2d(2, 2, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            (13): ReLU(inplace=True)
            (14): Conv2d(2, 2, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            (15): ReLU(inplace=True)
            (16): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
            (17): Conv2d(2, 2, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            (18): ReLU(inplace=True)
            (19): Conv2d(2, 2, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            (20): ReLU(inplace=True)
            (21): Conv2d(2, 2, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            (22): ReLU(inplace=True)
            (23): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
            (24): Conv2d(2, 2, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            (25): ReLU(inplace=True)
            (26): Conv2d(2, 2, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            (27): ReLU(inplace=True)
            (28): Conv2d(2, 1, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            (29): ReLU(inplace=True)
            (30): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
        )
    )
    ```