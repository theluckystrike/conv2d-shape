# conv2d-shape

Compute Conv2d output spatial size, height/width pairs, and trainable
parameter count from input size, kernel size, stride and padding — the
same formula PyTorch's `nn.Conv2d`, TensorFlow's `Conv2D` and ONNX runtimes
apply internally.

Worked examples for common input/kernel/stride combinations, with PyTorch
code and memory-footprint notes, are at
[HeyTensor's Conv2d answer pages](https://heytensor.com/answers/conv2d-output-112x112-kernel-3-stride-2.html).

## Usage

```go
import "github.com/theluckystrike/conv2d-shape"

out := conv2dshape.Output(112, 3, 2, 1)       // 56
h, w := conv2dshape.OutputHW(112, 64, 3, 2, 1) // 56, 32
p := conv2dshape.Params(64, 128, 3)            // 73856
```

License: MIT
