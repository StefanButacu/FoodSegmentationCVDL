import napari
import numpy as np

from models.unet_from_scratch.pilutil import bytescale


#
def re_normalize(inp: np.ndarray, low: int = 0, high: int = 255):
    """Normalize the data to a certain range. Default: [0-255]"""
    inp_out = bytescale(inp, low=low, high=high)
    return inp_out


def enable_gui_qt():
    """Performs the magic command %gui qt"""
    from IPython import get_ipython

    ipython = get_ipython()
    ipython.magic("gui qt")


class DatasetViewer:
    def __init__(self, dataset):

        self.dataset = dataset
        self.index = 0

        # napari viewer instance
        self.viewer = None

        # current image & shape layer
        self.image_layer = None
        self.label_layer = None

    def napari(self):
        # IPython magic for napari < 0.4.8
        enable_gui_qt()

        # napari
        if self.viewer:
            try:
                del self.viewer
            except AttributeError:
                pass
        self.index = 0

        # Init napari instance
        self.viewer = napari.Viewer()

        # Show current sample
        self.show_sample()

        # Key-bindings
        # Press 'n' to get the next sample
        @self.viewer.bind_key("n")
        def next(viewer):
            self.increase_index()  # Increase the index
            self.show_sample()  # Show next sample

        # Press 'b' to get the previous sample
        @self.viewer.bind_key("b")
        def prev(viewer):
            self.decrease_index()  # Decrease the index
            self.show_sample()  # Show next sample

    def increase_index(self):
        self.index += 1
        if self.index >= len(self.dataset):
            self.index = 0

    def decrease_index(self):
        self.index -= 1
        if self.index < 0:
            self.index = len(self.dataset) - 1

    def show_sample(self):

        # Get a sample from the dataset
        sample = self.get_sample_dataset(self.index)
        x, y = sample

        # Get the names from the dataset
        names = self.get_names_dataset(self.index)
        x_name, y_name = names
        x_name, y_name = x_name.name, y_name.name  # only possible if pathlib.Path

        # Transform the sample to numpy, cpu and correct format to visualize
        x = self.transform_x(x)
        y = self.transform_y(y)

        # Create or update image layer
        if self.image_layer not in self.viewer.layers:
            self.image_layer = self.create_image_layer(x, x_name)
        else:
            self.update_image_layer(self.image_layer, x, x_name)

        # Create or update label layer
        if self.label_layer not in self.viewer.layers:
            self.label_layer = self.create_label_layer(y, y_name)
        else:
            self.update_label_layer(self.label_layer, y, y_name)

        # Reset view
        self.viewer.reset_view()

    def create_image_layer(self, x, x_name):
        return self.viewer.add_image(x, name=str(x_name))

    def update_image_layer(self, image_layer, x, x_name):
        """Replace the data and the name of a given image_layer"""
        image_layer.data = x
        image_layer.name = str(x_name)

    def create_label_layer(self, y, y_name):
        return self.viewer.add_labels(y, name=str(y_name))

    def update_label_layer(self, target_layer, y, y_name):
        """Replace the data and the name of a given image_layer"""
        target_layer.data = y
        target_layer.name = str(y_name)

    def get_sample_dataset(self, index):
        return self.dataset[index]

    def get_names_dataset(self, index):
        return self.dataset.inputs[index], self.dataset.targets[index]

    def transform_x(self, x):
        # make sure it's a numpy.ndarray on the cpu
        x = x.cpu().numpy()

        # from [C, H, W] to [H, W, C] - only for RGB images.
        if self.check_if_rgb(x):
            x = np.moveaxis(x, source=0, destination=-1)

        # Re-normalize
        x = re_normalize(x)

        return x

    def transform_y(self, y):
        # make sure it's a numpy.ndarray on the cpu
        y = y.cpu().numpy()

        return y

    def check_if_rgb(self, x):
        return True if x.shape[0] == 3 else False


class OutputViewer:
    def __init__(self, inputs, target, predictions):

        self.inputs = inputs
        self.target = target
        self.predictions = predictions
        self.index = 0

        # napari viewer instance
        self.viewer = None

        # current image & shape layer
        self.image_layer = None
        self.label_layer = None

        self.prediction_layer = None

    def napari(self):
        # IPython magic for napari < 0.4.8
        enable_gui_qt()

        # napari
        if self.viewer:
            try:
                del self.viewer
            except AttributeError:
                pass
        self.index = 0

        # Init napari instance
        self.viewer = napari.Viewer()

        # Show current sample
        self.show_sample()

        # Key-bindings
        # Press 'n' to get the next sample
        @self.viewer.bind_key("n")
        def next(viewer):
            print("Next")
            self.increase_index()  # Increase the index
            self.show_sample()  # Show next sample

        # Press 'b' to get the previous sample
        @self.viewer.bind_key("b")
        def prev(viewer):
            self.decrease_index()  # Decrease the index
            self.show_sample()  # Show next sample

    def increase_index(self):
        self.index += 1
        if self.index >= len(self.dataset):
            self.index = 0

    def decrease_index(self):
        self.index -= 1
        if self.index < 0:
            self.index = len(self.dataset) - 1

    def show_sample(self):

        # Get a sample from the dataset
        sample = self.get_sample_dataset(self.index)
        x, y, out = sample
        # Get the names from the dataset
        x_name, y_name = "Input", "Target"  # only possible if pathlib.Path

        # Transform the sample to numpy, cpu and correct format to visualize
        x = self.transform_x(x)
        y = self.transform_y(y)
        out = self.transform_y(out)

        # Create or update image layer
        if self.image_layer not in self.viewer.layers:
            self.image_layer = self.create_image_layer(x, x_name)
        else:
            self.update_image_layer(self.image_layer, x, x_name)

        # Create or update label layer
        if self.label_layer not in self.viewer.layers:
            self.label_layer = self.create_label_layer(y, y_name)
        else:
            self.update_label_layer(self.label_layer, y, y_name)

        if self.prediction_layer not in self.viewer.layers:
            self.prediction_layer = self.create_prediction_layer(out, "Prediction")
        else:
            self.update_prediction_layer(self.prediction_layer, out, "Prediction")

        # Reset view
        self.viewer.reset_view()

    def create_prediction_layer(self, out, param):
        return self.viewer.add_labels(out, name="Prediction")

    def update_prediction_layer(self, prediction_layer, out, out_name):
        """Replace the data and the name of a given image_layer"""
        prediction_layer.data = out
        prediction_layer.name = "Output"

    def create_image_layer(self, x, x_name):
        return self.viewer.add_image(x, name="Input")

    def update_image_layer(self, image_layer, x, x_name):
        """Replace the data and the name of a given image_layer"""
        image_layer.data = x
        image_layer.name = "Input"

    def create_label_layer(self, y, y_name):
        return self.viewer.add_labels(y, name=str(y_name))

    def update_label_layer(self, target_layer, y, y_name):
        """Replace the data and the name of a given image_layer"""
        target_layer.data = y
        target_layer.name = str(y_name)

    def get_sample_dataset(self, index):
        return self.inputs[index], self.target[index].astype(np.uint8), self.predictions[index]

    def get_names_dataset(self, index):
        return ('input', 'target')

    def transform_x(self, x):
        # make sure it's a numpy.ndarray on the cpu
        # x = x.cpu().numpy()
        #
        # # from [C, H, W] to [H, W, C] - only for RGB images.
        if self.check_if_rgb(x):
            x = np.moveaxis(x, source=0, destination=-1)

        # # Re-normalize
        x = re_normalize(x)

        return x

    def transform_y(self, y):
        # make sure it's a numpy.ndarray on the cpu
        # y = y.cpu().numpy()
        return y

    def check_if_rgb(self, x):
        return True if x.shape[0] == 3 else False
