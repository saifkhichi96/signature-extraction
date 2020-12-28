# Extracting Signatures from Bank Checks

## Introduction
Ahmed et al. [1] suggest a connected components-based method for segmenting signatures in document images. For this purpose, they train and evaluate their models on Tobacco-800 documents dataset to extract patch-level signatures.

In this project, we have taken their proposal and modified it to work specifically for bank checks, and be able to segment signatures on a stroke level instead of patch level. Both these are inherently more difficult problems.

Bank checks have complex backgrounds which are not known beforehand. This was not the case in Tobacco-800 dataset where the documents have a simple white background. Similarly, extracting stroke-level position of the signature is again a more difficult task compared to patch level location.

## Demo
You can watch a demo of the results in the following [Youtube video](https://www.youtube.com/watch?v=mSPeYTF9J4Q)<br>
[![Demo Video](https://img.youtube.com/vi/mSPeYTF9J4Q/0.jpg)](https://www.youtube.com/watch?v=mSPeYTF9J4Q)

## Datasets
One reason why [1] worked on patch-level was a lack of publicly available datasets with groundtruth at stroke-level. In this project, we have created our own dataset of bank checks with stroke level groundtruth available as binary segmentation masks. We use this dataset for testing.

We divide our training data into two classes: `signatures` and `non-signature`. For `non-signature` class, we took 64 documents from Tobacco-800 [3, 4] to manually generate new images which which didn't contain any signatures or handwritten text. Some of these documents contained logos, as well. For the `signature` class, we combined all images from the UTSig signature database [2], which contains 8280 Persian language signatures.

## Installation
Clone the project and `cd` to project directory. Then, execute the following commands once to set-up your environment
```
python3 -m venv venv/
source venv/bin/activate
pip install -r requirements.txt
```

Then you can run the GUI app with `python3 src/extract_gui.py`.

The repository comes with trained models in the `models/` folder. To train a new model with your own data, look at the [train_test.py](./src/train_test.py) script.

## References
[1] Ahmed, S., Malik, M. I., Liwicki, M., & Dengel, A. (2012, September). Signature segmentation from document images. In 2012 International Conference on Frontiers in Handwriting Recognition (pp. 425-429). IEEE.

[2] Soleimani, A., Fouladi, K., & Araabi, B. N. (2016). UTSig: A Persian offline signature dataset. IET Biometrics, 6(1), 1-8.

[3] Zhu, G., & Doermann, D. (2007, September). Automatic document logo detection. In Ninth International Conference on Document Analysis and Recognition (ICDAR 2007) (Vol. 2, pp. 864-868). IEEE.

[4] Zhu, G., Zheng, Y., Doermann, D., & Jaeger, S. (2007, June). Multi-scale structural saliency for signature detection. In 2007 IEEE Conference on Computer Vision and Pattern Recognition (pp. 1-8). IEEE.
