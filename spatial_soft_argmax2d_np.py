import numpy as np


def create_meshgrid(x, normalized_coordinates):
    assert len(x.shape) == 4, x.shape
    _, _, height, width = x.shape
    if normalized_coordinates:
        xs = np.linspace(-1.0, 1.0, width)
        ys = np.linspace(-1.0, 1.0, height)
    else:
        xs = np.linspace(0, width - 1, width)
        ys = np.linspace(0, height - 1, height)
    return np.meshgrid(ys, xs)  # pos_y, pos_x


class SpatialSoftArgmax2d:
    def __init__(self, normalized_coordinates=True):
        self.normalized_coordinates = normalized_coordinates
        self.eps = 1e-6

    def forward(self, input):
        if not isinstance(input, np.ndarray):
            raise TypeError("Input input type is not a np.ndarray. Got {}"
                            .format(type(input)))
        if not len(input.shape) == 4:
            raise ValueError("Invalid input shape, we expect BxCxHxW. Got: {}"
                             .format(input.shape))
        # unpack shapes and create view from input tensor
        batch_size, channels, height, width = input.shape
        x = input.reshape(batch_size, channels, -1)

        # compute softmax with max subtraction trick
        exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        exp_x_sum = 1.0 / (np.sum(exp_x, axis=-1, keepdims=True) + self.eps)

        # create coordinates grid
        pos_x, pos_y = create_meshgrid(input, self.normalized_coordinates)
        pos_x = pos_x.reshape(-1)
        pos_y = pos_y.reshape(-1)

        # compute the expected coordinates
        expected_y = np.sum((pos_y * exp_x) * exp_x_sum, axis=-1, keepdims=True)
        expected_x = np.sum((pos_x * exp_x) * exp_x_sum, axis=-1, keepdims=True)
        output = np.concatenate([expected_x, expected_y], axis=-1)
        return output.reshape(batch_size, channels, 2)  # BxNx2

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)
