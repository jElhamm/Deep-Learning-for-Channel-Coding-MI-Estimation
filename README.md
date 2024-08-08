# Deep Learning for Channel Coding via Neural Mutual Information Estimation

   This repository contains the implementation of the paper [**"Deep Learning for Channel Coding via Neural Mutual Information Estimation"**](https://github.com/jElhamm/Deep-Learning-for-Channel-Coding-MI-Estimation/blob/main/Deep%20Learning%20for%20Channel%20Coding%20via%20Neural.pdf).

   The paper presents novel approaches for channel coding using neural networks, focusing on estimating [Mutual Information (MI)](https://en.wikipedia.org/wiki/Mutual_information) to optimize the performance of autoencoders under Rayleigh and Binary Input channels.
   

## Citation

   If you find this repository useful in your research, please consider citing the following paper:

```bibtex
   @article{fritschek2019deep,
      title={Deep Learning for Channel Coding via Neural Mutual Information Estimation},
      author={Rick Fritschek, Rafael F. Schaefery, and Gerhard Wunder},
      journal={IEEE Transactions on Information Theory},
      volume={65},
      number={4},
      pages={2042--2059},
      year={2019},
      publisher={IEEE}
   }
```

## Table of Contents
   - [Overview](#overview)
   - [File Descriptions](#file-descriptions)
   - [Installation](#installation)
   - [Usage](#usage)
      - [Python Scripts](#python-scripts)
      - [Jupyter Notebooks](#jupyter-notebooks)
   - [Results](#results)
   - [Requirements](#requirements)
   - [Contributing](#contributing)
   - [License](#license)

## Overview

   This repository includes:

   * The original research paper in PDF format.
   * Python implementations of the models and experiments discussed in the paper.
   * Jupyter Notebooks for interactive exploration of the models.

   The primary focus is on using *Mutual Information Neural Estimation (MINE)* to enhance the performance of autoencoders in channel coding tasks.
   The code has been structured to facilitate ease of understanding and experimentation.

## File Descriptions

### [Python Scripts](Source%20Code/Python%20Sources%20File)

   | Filename | Description |
   | -------- | ----------- |
   | [`AutoEncoder_Rayleigh_Channel_MINE.py`](Python%20Sources%20File/AutoEncoder_Rayleigh_Channel_MINE.py) | Implements the autoencoder architecture for Rayleigh channels with MI estimation using MINE. |
   | [`Autoencoder_MINE_for_Binary_Input.py`](Python%20Sources%20File/Autoencoder_MINE_for_Binary%20Input.py) | Implements the autoencoder architecture optimized for binary input using MI estimation. |
   | [`MINE_Encoder_Experimental_Model.py`](Python%20Sources%20File/MINE_Encoder_Experimental_Model.py) | Contains experimental models for testing the performance of MINE-based encoders under various channel conditions. |

### [Jupyter Notebooks](Source%20Code/Jupyter%20Notebook%20Source%20File)

   | Filename | Description |
   | -------- | ----------- |
   | [`AutoEncoder_Rayleigh_Channel_MINE.ipynb`](Jupyter%20Notebook%20Source%20File/AutoEncoder_Rayleigh_Channel_MINE.ipynb) | Interactive version of the Python script for Rayleigh channels, with step-by-step explanations and visualizations. |
   | [`Autoencoder_MINE_for_Binary_Input.ipynb`](Jupyter%20Notebook%20Source%20File/Autoencoder_MINE_for_Binary_Input.ipynb) | Interactive version of the Python script optimized for binary input, allowing for parameter adjustments and result visualization. |
   | [`MINE_Encoder_Experimental_Model.ipynb`](Jupyter%20Notebook%20Source%20File/MINE_Encoder_Experimental_Model.ipynb) | Interactive notebook for experimenting with different models and channel conditions, demonstrating the efficacy of the MINE approach. |

## Installation

1. **Clone the repository:**
    ```bash
    git clone https://github.com/jElhamm/Deep-Learning-for-Channel-Coding-MI-Estimation
    cd Deep-Learning-for-Channel-Coding-via-Neural-Mutual-Information-Estimation
    ```

2. **Install the required packages:**
    ```bash
    pip install -r requirements.txt
    ```

3. **Open Jupyter Notebooks:**
    ```bash
    jupyter notebook
    ```

## Usage

   To run the Python scripts directly:
```bash
   python AutoEncoder_Rayleigh_Channel_MINE.py
   python Autoencoder_MINE_for_Binary_Input.py
   python MINE_Encoder_Experimental_Model.py
```

   Alternatively, you can explore the Jupyter Notebooks to interactively execute and visualize the results.

## Results

   The results of the experiments conducted using these scripts and notebooks demonstrate the effectiveness of using MINE for channel coding tasks.
   The implementation shows significant improvements in *Bit Error Rates (BER)* compared to traditional methods, particularly in noisy channel conditions.
   Visualizations and detailed results are available within the respective Jupyter Notebooks.

## Requirements

   This project requires the following Python libraries:

   | Library     | Version | Purpose                                           |
   |-------------|---------|---------------------------------------------------|
   |  numpy      | 1.26.4  | Used for numerical computations and array manipulations. |
   |  scipy      | 1.13.1  | Used for scientific and technical computations. |
   |  keras      | 3.4.1   | Used for building and training deep learning models. |
   |  tensorflow | 2.17.0  | Provides the backend for deep learning models and computations. |
   |  matplotlib | 3.7.1   | Used for creating visualizations and plots. |

To install these dependencies, you can use the [requirements.txt](requirements.txt) file included in the repository. Run the following command:

```bash
pip install -r requirements.txt
```

## Contributing

   We welcome contributions to this project! To contribute, please:

   1. Fork the repository and clone your fork.
   2. Create a new branch for your changes.
   3. Make and test your changes, then commit them.
   4. Push your changes to your fork and submit a pull request.

## License

   This repository is licensed under the BSD-3-Clause License.
   See the [LICENSE](./LICENSE) file for more details.