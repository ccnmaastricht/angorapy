# Dexterous Robot Hand

## Usage

### Requirements

This project was scripted and tested for `python 3.6` and should also work on any newer version of python.
Due to heavy usage of type hinting the code is not compatible with any older version of python!

To install all required python packages run

```bash
pip install -r requirements.txt
```

Additionally some libraries and permissions will be required. Run

```bash
sudo bash install.sh
```

To train on any mujoco-based environment, you will need to install the MuJoCo software (requiring a license) as well as mujoco-py.

To save gifs during training, imagemagicks policy needs to be changed to allow more memory usage. Achieve this by e.g. commenting out all ressource lines in `/etc/ImageMagick-6/policy.xml`. 

### Main Scripts

The python files `train.py` and `evaluate.py` provide ready-made scripts for training 
and evaluating an agent in an environment. With `pretrain.py`, it is possible to pretrain the visual component
of a network on classification or reconstruction.

## Visualization

The `analysis` module provides the `NetworkAnalyzer` class, which can be used to visualize certain aspects of a trained 
model. For any existing model, the Analyzer simply functions as a wrapper for the model:

```python
analyzer = NetworkAnalyzer(model, mode="show")
```

where the `mode` parameter can be either _show_ or _save_, resulting in a popup or file saving of visualizations 
respectively. Saved visualizations will be located in _docs/analysis/figures_ with descriptive filenames. The analyzer 
can alternatively also be initialized from a saved model as follows:

```python
analyzer = NetworkAnalyzer.from_saved_model("path/to/my/model", mode="show")
```

Most visuzalization methods require naming the layer of interest. The layer is identified by its unique name.
A list of layer names of the analyzed model can be retrieved using `list_layer_names()`, 
`list_convolutional_layer_names()` or `list_non_convolutional_layer_names()`.

Currently, the following visualization methods are supported.

### Weight Visualization
**Weight Visualization** simply interprets the weights of every filter in a layer as an image. This requires the layers input channels 
to be either 1 or 3. For the first convolutional layer of a pretrained VGG16 model, this can be used as follows:

```python
model = tf.keras.applications.VGG16()
analyzer = NetworkAnalyzer(model, mode="show")
analyzer.visualize_layer_weights("block1_conv1")
```

This produces the following visualization:

![](https://i.postimg.cc/P5fCYQrx/weights-block1-conv1.png)

### Feature Maximization
**Feature Maximization** creates an input image from random noise and optimizes its pixels to maximize the response of
(a specific) filter(s) in a layer, using gradient descent. To make the result more interpretable, the process starts at 
a low input resolution and successively upscales the input. This produces patterns of lower frequency. 

For the first convolutional layer of a pretrained VGG16 model, this can be used as follows:
```python
analyzer.visualize_max_filter_respondence("block2_conv2", feature_ids=[24, 57, 89, 120, 42, 26, 45, 21, 99])
```
![](https://i.postimg.cc/6QgTssV2/feature-maximization-block2-conv2-24-57-89-120-42-26-45-21-99.png)

### Feature Maps
**Feature Maps** take a given input image and record the activation at a specified layer. These activations can then 
be visualized in three ways, specified through a mode.

#### Gray Image Mode
In the default gray image mode (`mode="gray"`) the activation map is simply used like an image. For layers
with spacial dimensions close to the input, this gives spatially precise info, but deeper into the network,
these images get smaller and smaller and less interpretable.

```python
reference = mpimg.imread("hand.png")
analyzer.visualize_activation_map("block1_conv1", reference, mode="gray")
```

![](https://i.postimg.cc/sf92rL46/feature-maps-block1-conv1-gray.png)

#### Heatmap Mode
To tackle the issue with large receptive fields in deeper layers, the heatmap mode (`mode="heat"`) converts the
activation into a heatmap contour plot and scales it to the original image size. The resulting plot is the original
image in greyscale, overlayed by this heatmap.

```python
reference = mpimg.imread("hand.png")
analyzer.visualize_activation_map("block1_conv1", reference, mode="heat")
```

![](https://i.postimg.cc/ZnL2Mp3H/feature-maps-block1-conv1-heat.png)

#### Bar Plot Mode
Lastly, the bar plot mode (`mode="plot"`) shows the activation of a filter as a bar in a bar plot, averaged over the 
spatial dimensions.

```python
reference = mpimg.imread("hase.jpg")
analyzer.visualize_activation_map("block1_conv2", reference, mode="gray")
```

![](https://i.postimg.cc/DyPWSMhH/feature-maps-block1-conv2-plot.png)
