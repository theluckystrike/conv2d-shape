// Package conv2dshape computes the spatial output dimensions of a 2D
// convolution layer from its input size, kernel size, stride and padding.
//
// The formula is the one every deep-learning framework (PyTorch's
// nn.Conv2d, TensorFlow's Conv2D, ONNX) implements internally:
// output = floor((input + 2*padding - kernel) / stride) + 1, applied
// independently to height and width. Worked examples for common
// input/kernel/stride combinations, with PyTorch code, are at
// https://heytensor.com/answers/conv2d-output-112x112-kernel-3-stride-2.html
// and the sibling pages linked from there.
//
// Params covers a single Conv2d layer in isolation. To compose the per-layer
// counts this package returns into a whole-network parameter, FLOPs and
// memory total - mixing Conv2D with Dense, LSTM and multi-head attention
// layers, and checking the result against ResNet, VGG, BERT and GPT
// baselines - see
// https://ml0x.com/tools/model-complexity-calculator.html
package conv2dshape

// Output returns the output spatial size of a single Conv2d dimension
// (height or width) for the given input size, kernel size, stride and
// padding. Dilation is assumed to be 1. Returns 0 for a non-positive
// stride or a kernel/padding combination that would produce a negative
// receptive field.
func Output(input, kernel, stride, padding int) int {
	if stride <= 0 {
		return 0
	}
	n := input + 2*padding - kernel
	if n < 0 {
		return 0
	}
	return n/stride + 1
}

// OutputHW returns the (height, width) output shape for a possibly
// non-square input, applying Output independently to each dimension.
func OutputHW(inH, inW, kernel, stride, padding int) (int, int) {
	return Output(inH, kernel, stride, padding), Output(inW, kernel, stride, padding)
}

// Params returns the trainable parameter count of a Conv2d layer:
// in_channels * out_channels * kernel^2 + out_channels (bias term).
func Params(inChannels, outChannels, kernel int) int {
	return inChannels*outChannels*kernel*kernel + outChannels
}
