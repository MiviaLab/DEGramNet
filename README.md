# DEGramNet: Effective audio analysis based on a fully learnable time-frequency representation

![](https://img.shields.io/badge/keras-tensorflow-blue.svg)

This repository is the official implementation of [DEGramNet: Effective audio analysis based on a fully learnable time-frequency representation](https://link.springer.com/article/10.1007/s00521-023-08849-7). 

For more information you can contact the authors at: pfoggia@unisa.it, agreco@unisa.it, roberto.antonio@outlook.it, asaggese@unisa.it, mvento@unisa.it .

<img src='./image.png' align='center' width=600 style="margin-left:3%;margin-bottom:3%">

### Citations

If you use this code in your research, please cite these papers.


```bibtext
@article{foggia2023degramnet,
  title={Degramnet: effective audio analysis based on a fully learnable time-frequency representation},
  author={Foggia, Pasquale and Greco, Antonio and Roberto, Antonio and Saggese, Alessia and Vento, Mario},
  journal={Neural Computing and Applications},
  pages={1--13},
  year={2023},
  publisher={Springer}
}

@article{greco2021denet,
  title={DENet: a deep architecture for audio surveillance applications},
  author={Greco, Antonio and Roberto, Antonio and Saggese, Alessia and Vento, Mario},
  journal={Neural Computing and Applications},
  doi={10.1007/s00521-020-05572-5},
  pages={1--12},
  year={2021},
  publisher={Springer}
}
```

DEGramNet is an innovative convolutional architecture for audio analysis tasks, addressing the limitations of current state-of-the-art algorithms. Unlike traditional hand-crafted Spectrogram-like representations, DEGramNet utilizes a novel, compact, and trainable time-frequency representation called DEGram. This representation overcomes the drawbacks of fixed filter parameters by dynamically learning the frequencies of interest specific to the audio analysis task. DEGramNet incorporates a custom time-frequency attention module within the DEGram representation, enabling it to perform denoising on the audio signal in both the time and frequency domains. By amplifying the relevant frequency and time components of the sound, DEGramNet effectively improves the generalization capabilities, especially when training data is limited. Moreover, this flexibility allows the representation to adapt easily to different audio analysis problems, such as emphasizing voice frequencies for speaker recognition.

## Requirements

To install the requirements:

```bash
git clone https://github.com/MiviaLab/DEGramNet.git
cd DEGramNet
pip install -r requirements.txt         # with CUDA
pip install -r requirements-nogpu.txt   # without CUDA
```

The docker files are also available for reproducibility purposes.

## Usage

```python
get_degramnet(input_shape, window_order=4, attention=True, znorm_freq=False, get_full_model=True, n_classes=100, add_dropout=0.5,)
``` 

- *input_shape*: tuple in the form (samples, 1)
- *window_order*: Butterworth window order  
- *attention*: True to use DEGram, False to use SincGram  
- *znorm_freq*: True to scale spectrogram features after mean removal, False instead  
- *get_full_model*: To get both the representation and classification layers, False to get only the representation ones  
- *n_classes*: Size of the prediction layer (ignored if get_full_model is False)  
- *dropout*: dropout probability for the Dropout layer befor the prediction one, 0.0 to skip

## Example

```python
import numpy as np
from degramnet import get_degramnet

# 10 seconds of audio sampled at 16KHz
input_shape = (160000,1)

# get the model
model = get_degramnet(
    input_shape,
    window_order=4,
    attention=True,
    znorm_freq=False,
    get_full_model=True,
    n_classes=100,
    add_dropout=0.5,
)

# Print the model 
model.summary()

# Predict random data
X = np.random.rand(1,*input_shape)
y = model.predict(X)

print(y.shape)
```

### License
The code and mode are available to download for commercial/research purposes under a Creative Commons Attribution 4.0 International License(https://creativecommons.org/licenses/by/4.0/).

      Downloading this code implies agreement to follow the same conditions for any modification 
      and/or re-distribution of the dataset in any form.

      Additionally any entity using this code agrees to the following conditions:

      THIS CODE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS
      IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED
      TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
      PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
      HOLDER BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
      EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
      PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
      PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
      LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
      NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
      SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

      Please cite the paper if you make use of the dataset and/or code.
