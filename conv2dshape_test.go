package conv2dshape

import "testing"

func TestOutput112(t *testing.T) {
	if got := Output(112, 3, 2, 1); got != 56 {
		t.Fatalf("Output(112,3,2,1) = %d, want 56", got)
	}
}

func TestOutputHWSquare(t *testing.T) {
	h, w := OutputHW(224, 224, 3, 2, 1)
	if h != 112 || w != 112 {
		t.Fatalf("OutputHW(224,224,3,2,1) = (%d,%d), want (112,112)", h, w)
	}
}

func TestOutputHWNonSquare(t *testing.T) {
	h, w := OutputHW(112, 64, 3, 2, 1)
	if h != 56 || w != 32 {
		t.Fatalf("OutputHW(112,64,3,2,1) = (%d,%d), want (56,32)", h, w)
	}
}

func TestParams(t *testing.T) {
	if got := Params(64, 128, 3); got != 73856 {
		t.Fatalf("Params(64,128,3) = %d, want 73856", got)
	}
}

func TestOutputZeroStride(t *testing.T) {
	if got := Output(10, 3, 0, 0); got != 0 {
		t.Fatalf("Output with zero stride = %d, want 0", got)
	}
}

func TestOutputSamePadding(t *testing.T) {
	// stride=1, padding=(kernel-1)/2 must preserve spatial size
	if got := Output(56, 3, 1, 1); got != 56 {
		t.Fatalf("same-padding Output(56,3,1,1) = %d, want 56", got)
	}
}
