# 实验5_TFLite模型生成
## 1 了解机器学习基础
1. 什么是机器学习？\
    	**机器学习定义（Machine Learning definition）**
	​	Arthur Samu（1959）：在没有明确设置的情况下，是计算机具有学习能力的研究领域。

​	Tom Mitchell（1998）：计算机程序从经验E中学习解决某一任务T进行某一性能度量P，通过P测定T上的表现因经验E而提高。
​	例如：跳棋游戏，经验E就是程序和自己下几万次跳棋，任务T就是玩跳棋，性能度量P就是与新对手玩跳棋时赢的概率。

2. **监督学习（Supervised learning）**
	监督学习定义：简单来说，我们会教计算机做某件事。
	有输入数据和输出数据通过学习训练集中输入数据和输出数据的关系，生成合适的函数，将输入映射到合适的输出。比如分类、回归。

3. **无监督学习（Unsupervised learning）**
	无监督学习定义：简单来说，我们让计算机自己学习。
	无监督学习是一种机器学习的训练方式，它本质上是一个统计手段，在没有标签的数据里可以发现潜在的一些结构的一种训练方式。

​	它主要具备3个特点：
​	①无监督学习没有明确的目的
​	②无监督学习不需要给数据打标签
​	③无监督学习无法量化效果

4. **半监督学习**
半监督学习：定义训练集中一部分数据有特征和标签，另一部分只有特征（只有输入没有输出），综合两类数据来生成合适的函数。

## 2 了解TensorFlow及TensorFlow Lite
- TensorFlow：一个核心开源库，可以快速的开发和训练机器学习模型。通过直接在浏览器中运行 Colab 笔记本来快速上手。 
- TensorFlow = Tensor+Flow。Tensor代表“张量”，是一种多维数组结构；Flow代表“流”，TensorFlow采用数据流的进行处理。
- TensorFlow Lite是移动机器学习库，用于在移动设备、微控制器以及其他终端设备上部署模型。

## 3 按照教程完成基于TensorFlow Lite Model Maker的花卉模型生成
### 使用TensorFlow Lite Model Maker生成图像分类器模型

#### 预备工作
首先安装程序运行必备的一些库。


```python
!pip install -i https://pypi.tuna.tsinghua.edu.cn/simple/ tflite-model-maker --user
```

    Looking in indexes: https://pypi.tuna.tsinghua.edu.cn/simple/
    Requirement already satisfied: tflite-model-maker in c:\users\16588\appdata\roaming\python\python39\site-packages (0.3.4)
    Requirement already satisfied: tf-models-official==2.3.0 in c:\users\16588\appdata\roaming\python\python39\site-packages (from tflite-model-maker) (2.3.0)
    Requirement already satisfied: pillow>=7.0.0 in d:\anaconda3\lib\site-packages (from tflite-model-maker) (9.0.1)
    Requirement already satisfied: tensorflow-datasets>=2.1.0 in c:\users\16588\appdata\roaming\python\python39\site-packages (from tflite-model-maker) (4.5.2)
    Requirement already satisfied: tflite-support>=0.3.1 in c:\users\16588\appdata\roaming\python\python39\site-packages (from tflite-model-maker) (0.4.0)
    Requirement already satisfied: numpy>=1.17.3 in d:\anaconda3\lib\site-packages (from tflite-model-maker) (1.21.5)
    Requirement already satisfied: tensorflowjs>=2.4.0 in c:\users\16588\appdata\roaming\python\python39\site-packages (from tflite-model-maker) (3.18.0)
    Requirement already satisfied: Cython>=0.29.13 in d:\anaconda3\lib\site-packages (from tflite-model-maker) (0.29.28)
    Requirement already satisfied: PyYAML>=5.1 in d:\anaconda3\lib\site-packages (from tflite-model-maker) (6.0)
    Requirement already satisfied: matplotlib<3.5.0,>=3.0.3 in c:\users\16588\appdata\roaming\python\python39\site-packages (from tflite-model-maker) (3.4.3)
    Requirement already satisfied: tensorflow-addons>=0.11.2 in c:\users\16588\appdata\roaming\python\python39\site-packages (from tflite-model-maker) (0.17.0)
    Requirement already satisfied: lxml>=4.6.1 in d:\anaconda3\lib\site-packages (from tflite-model-maker) (4.8.0)
    Requirement already satisfied: tensorflow-model-optimization>=0.5 in c:\users\16588\appdata\roaming\python\python39\site-packages (from tflite-model-maker) (0.7.2)
    Requirement already satisfied: tensorflow>=2.6.0 in c:\users\16588\appdata\roaming\python\python39\site-packages (from tflite-model-maker) (2.9.1)
    Requirement already satisfied: absl-py>=0.10.0 in c:\users\16588\appdata\roaming\python\python39\site-packages (from tflite-model-maker) (1.0.0)
    Requirement already satisfied: tensorflow-hub<0.13,>=0.7.0 in c:\users\16588\appdata\roaming\python\python39\site-packages (from tflite-model-maker) (0.12.0)
    Requirement already satisfied: flatbuffers==1.12 in c:\users\16588\appdata\roaming\python\python39\site-packages (from tflite-model-maker) (1.12)
    Requirement already satisfied: fire>=0.3.1 in c:\users\16588\appdata\roaming\python\python39\site-packages (from tflite-model-maker) (0.4.0)
    Requirement already satisfied: librosa==0.8.1 in c:\users\16588\appdata\roaming\python\python39\site-packages (from tflite-model-maker) (0.8.1)
    Requirement already satisfied: sentencepiece>=0.1.91 in c:\users\16588\appdata\roaming\python\python39\site-packages (from tflite-model-maker) (0.1.96)
    Requirement already satisfied: urllib3!=1.25.0,!=1.25.1,<1.26,>=1.21.1 in c:\users\16588\appdata\roaming\python\python39\site-packages (from tflite-model-maker) (1.25.11)
    Requirement already satisfied: neural-structured-learning>=1.3.1 in c:\users\16588\appdata\roaming\python\python39\site-packages (from tflite-model-maker) (1.3.1)
    Requirement already satisfied: numba==0.53 in c:\users\16588\appdata\roaming\python\python39\site-packages (from tflite-model-maker) (0.53.0)
    Requirement already satisfied: six>=1.12.0 in d:\anaconda3\lib\site-packages (from tflite-model-maker) (1.16.0)
    Requirement already satisfied: joblib>=0.14 in d:\anaconda3\lib\site-packages (from librosa==0.8.1->tflite-model-maker) (1.1.0)
    Requirement already satisfied: pooch>=1.0 in c:\users\16588\appdata\roaming\python\python39\site-packages (from librosa==0.8.1->tflite-model-maker) (1.6.0)
    Requirement already satisfied: decorator>=3.0.0 in d:\anaconda3\lib\site-packages (from librosa==0.8.1->tflite-model-maker) (5.1.1)
    Requirement already satisfied: packaging>=20.0 in c:\users\16588\appdata\roaming\python\python39\site-packages (from librosa==0.8.1->tflite-model-maker) (20.9)
    Requirement already satisfied: scipy>=1.0.0 in d:\anaconda3\lib\site-packages (from librosa==0.8.1->tflite-model-maker) (1.7.3)
    Requirement already satisfied: resampy>=0.2.2 in c:\users\16588\appdata\roaming\python\python39\site-packages (from librosa==0.8.1->tflite-model-maker) (0.2.2)
    Requirement already satisfied: soundfile>=0.10.2 in c:\users\16588\appdata\roaming\python\python39\site-packages (from librosa==0.8.1->tflite-model-maker) (0.10.3.post1)
    Requirement already satisfied: audioread>=2.0.0 in c:\users\16588\appdata\roaming\python\python39\site-packages (from librosa==0.8.1->tflite-model-maker) (2.1.9)
    Requirement already satisfied: scikit-learn!=0.19.0,>=0.14.0 in d:\anaconda3\lib\site-packages (from librosa==0.8.1->tflite-model-maker) (1.0.2)
    Requirement already satisfied: setuptools in d:\anaconda3\lib\site-packages (from numba==0.53->tflite-model-maker) (61.2.0)
    Requirement already satisfied: llvmlite<0.37,>=0.36.0rc1 in c:\users\16588\appdata\roaming\python\python39\site-packages (from numba==0.53->tflite-model-maker) (0.36.0)
    Requirement already satisfied: dataclasses in c:\users\16588\appdata\roaming\python\python39\site-packages (from tf-models-official==2.3.0->tflite-model-maker) (0.6)
    Requirement already satisfied: py-cpuinfo>=3.3.0 in c:\users\16588\appdata\roaming\python\python39\site-packages (from tf-models-official==2.3.0->tflite-model-maker) (8.0.0)
    Requirement already satisfied: gin-config in c:\users\16588\appdata\roaming\python\python39\site-packages (from tf-models-official==2.3.0->tflite-model-maker) (0.5.0)
    Requirement already satisfied: pandas>=0.22.0 in d:\anaconda3\lib\site-packages (from tf-models-official==2.3.0->tflite-model-maker) (1.4.2)
    Requirement already satisfied: google-api-python-client>=1.6.7 in c:\users\16588\appdata\roaming\python\python39\site-packages (from tf-models-official==2.3.0->tflite-model-maker) (2.49.0)
    Requirement already satisfied: opencv-python-headless in c:\users\16588\appdata\roaming\python\python39\site-packages (from tf-models-official==2.3.0->tflite-model-maker) (4.5.5.64)
    Requirement already satisfied: tf-slim>=1.1.0 in c:\users\16588\appdata\roaming\python\python39\site-packages (from tf-models-official==2.3.0->tflite-model-maker) (1.1.0)
    Requirement already satisfied: kaggle>=1.3.9 in c:\users\16588\appdata\roaming\python\python39\site-packages (from tf-models-official==2.3.0->tflite-model-maker) (1.5.12)
    Requirement already satisfied: psutil>=5.4.3 in d:\anaconda3\lib\site-packages (from tf-models-official==2.3.0->tflite-model-maker) (5.8.0)
    Requirement already satisfied: google-cloud-bigquery>=0.31.0 in c:\users\16588\appdata\roaming\python\python39\site-packages (from tf-models-official==2.3.0->tflite-model-maker) (3.1.0)
    Requirement already satisfied: termcolor in c:\users\16588\appdata\roaming\python\python39\site-packages (from fire>=0.3.1->tflite-model-maker) (1.1.0)
    Requirement already satisfied: kiwisolver>=1.0.1 in d:\anaconda3\lib\site-packages (from matplotlib<3.5.0,>=3.0.3->tflite-model-maker) (1.3.2)
    Requirement already satisfied: python-dateutil>=2.7 in d:\anaconda3\lib\site-packages (from matplotlib<3.5.0,>=3.0.3->tflite-model-maker) (2.8.2)
    Requirement already satisfied: pyparsing>=2.2.1 in d:\anaconda3\lib\site-packages (from matplotlib<3.5.0,>=3.0.3->tflite-model-maker) (3.0.4)
    Requirement already satisfied: cycler>=0.10 in d:\anaconda3\lib\site-packages (from matplotlib<3.5.0,>=3.0.3->tflite-model-maker) (0.11.0)
    Requirement already satisfied: attrs in d:\anaconda3\lib\site-packages (from neural-structured-learning>=1.3.1->tflite-model-maker) (21.4.0)
    Requirement already satisfied: wrapt>=1.11.0 in d:\anaconda3\lib\site-packages (from tensorflow>=2.6.0->tflite-model-maker) (1.12.1)
    Requirement already satisfied: opt-einsum>=2.3.2 in c:\users\16588\appdata\roaming\python\python39\site-packages (from tensorflow>=2.6.0->tflite-model-maker) (3.3.0)
    Requirement already satisfied: astunparse>=1.6.0 in c:\users\16588\appdata\roaming\python\python39\site-packages (from tensorflow>=2.6.0->tflite-model-maker) (1.6.3)
    Requirement already satisfied: keras-preprocessing>=1.1.1 in c:\users\16588\appdata\roaming\python\python39\site-packages (from tensorflow>=2.6.0->tflite-model-maker) (1.1.2)
    Requirement already satisfied: typing-extensions>=3.6.6 in d:\anaconda3\lib\site-packages (from tensorflow>=2.6.0->tflite-model-maker) (4.1.1)
    Requirement already satisfied: gast<=0.4.0,>=0.2.1 in c:\users\16588\appdata\roaming\python\python39\site-packages (from tensorflow>=2.6.0->tflite-model-maker) (0.4.0)
    Requirement already satisfied: tensorboard<2.10,>=2.9 in c:\users\16588\appdata\roaming\python\python39\site-packages (from tensorflow>=2.6.0->tflite-model-maker) (2.9.0)
    Requirement already satisfied: grpcio<2.0,>=1.24.3 in c:\users\16588\appdata\roaming\python\python39\site-packages (from tensorflow>=2.6.0->tflite-model-maker) (1.46.3)
    Requirement already satisfied: google-pasta>=0.1.1 in c:\users\16588\appdata\roaming\python\python39\site-packages (from tensorflow>=2.6.0->tflite-model-maker) (0.2.0)
    Requirement already satisfied: h5py>=2.9.0 in d:\anaconda3\lib\site-packages (from tensorflow>=2.6.0->tflite-model-maker) (3.6.0)
    Requirement already satisfied: keras<2.10.0,>=2.9.0rc0 in c:\users\16588\appdata\roaming\python\python39\site-packages (from tensorflow>=2.6.0->tflite-model-maker) (2.9.0)
    Requirement already satisfied: tensorflow-estimator<2.10.0,>=2.9.0rc0 in c:\users\16588\appdata\roaming\python\python39\site-packages (from tensorflow>=2.6.0->tflite-model-maker) (2.9.0)
    Requirement already satisfied: tensorflow-io-gcs-filesystem>=0.23.1 in c:\users\16588\appdata\roaming\python\python39\site-packages (from tensorflow>=2.6.0->tflite-model-maker) (0.26.0)
    Requirement already satisfied: protobuf<3.20,>=3.9.2 in d:\anaconda3\lib\site-packages (from tensorflow>=2.6.0->tflite-model-maker) (3.19.1)
    Requirement already satisfied: libclang>=13.0.0 in c:\users\16588\appdata\roaming\python\python39\site-packages (from tensorflow>=2.6.0->tflite-model-maker) (14.0.1)
    Requirement already satisfied: typeguard>=2.7 in c:\users\16588\appdata\roaming\python\python39\site-packages (from tensorflow-addons>=0.11.2->tflite-model-maker) (2.13.3)
    Requirement already satisfied: tqdm in d:\anaconda3\lib\site-packages (from tensorflow-datasets>=2.1.0->tflite-model-maker) (4.64.0)
    Requirement already satisfied: tensorflow-metadata in c:\users\16588\appdata\roaming\python\python39\site-packages (from tensorflow-datasets>=2.1.0->tflite-model-maker) (1.8.0)
    Requirement already satisfied: promise in c:\users\16588\appdata\roaming\python\python39\site-packages (from tensorflow-datasets>=2.1.0->tflite-model-maker) (2.3)
    Requirement already satisfied: requests>=2.19.0 in d:\anaconda3\lib\site-packages (from tensorflow-datasets>=2.1.0->tflite-model-maker) (2.27.1)
    Requirement already satisfied: dill in c:\users\16588\appdata\roaming\python\python39\site-packages (from tensorflow-datasets>=2.1.0->tflite-model-maker) (0.3.5.1)
    Requirement already satisfied: dm-tree~=0.1.1 in c:\users\16588\appdata\roaming\python\python39\site-packages (from tensorflow-model-optimization>=0.5->tflite-model-maker) (0.1.7)
    Requirement already satisfied: sounddevice>=0.4.4 in c:\users\16588\appdata\roaming\python\python39\site-packages (from tflite-support>=0.3.1->tflite-model-maker) (0.4.4)
    Requirement already satisfied: pybind11>=2.6.0 in c:\users\16588\appdata\roaming\python\python39\site-packages (from tflite-support>=0.3.1->tflite-model-maker) (2.9.2)
    Requirement already satisfied: wheel<1.0,>=0.23.0 in d:\anaconda3\lib\site-packages (from astunparse>=1.6.0->tensorflow>=2.6.0->tflite-model-maker) (0.37.1)
    Requirement already satisfied: httplib2<1dev,>=0.15.0 in c:\users\16588\appdata\roaming\python\python39\site-packages (from google-api-python-client>=1.6.7->tf-models-official==2.3.0->tflite-model-maker) (0.20.4)
    Requirement already satisfied: google-api-core!=2.0.*,!=2.1.*,!=2.2.*,!=2.3.0,<3.0.0dev,>=1.31.5 in c:\users\16588\appdata\roaming\python\python39\site-packages (from google-api-python-client>=1.6.7->tf-models-official==2.3.0->tflite-model-maker) (2.8.1)
    Requirement already satisfied: google-auth<3.0.0dev,>=1.16.0 in d:\anaconda3\lib\site-packages (from google-api-python-client>=1.6.7->tf-models-official==2.3.0->tflite-model-maker) (1.33.0)
    Requirement already satisfied: uritemplate<5,>=3.0.1 in c:\users\16588\appdata\roaming\python\python39\site-packages (from google-api-python-client>=1.6.7->tf-models-official==2.3.0->tflite-model-maker) (4.1.1)
    Requirement already satisfied: google-auth-httplib2>=0.1.0 in c:\users\16588\appdata\roaming\python\python39\site-packages (from google-api-python-client>=1.6.7->tf-models-official==2.3.0->tflite-model-maker) (0.1.0)
    Requirement already satisfied: pyarrow<9.0dev,>=3.0.0 in c:\users\16588\appdata\roaming\python\python39\site-packages (from google-cloud-bigquery>=0.31.0->tf-models-official==2.3.0->tflite-model-maker) (8.0.0)
    Requirement already satisfied: google-resumable-media<3.0dev,>=0.6.0 in d:\anaconda3\lib\site-packages (from google-cloud-bigquery>=0.31.0->tf-models-official==2.3.0->tflite-model-maker) (1.3.1)
    Requirement already satisfied: google-cloud-core<3.0.0dev,>=1.4.1 in c:\users\16588\appdata\roaming\python\python39\site-packages (from google-cloud-bigquery>=0.31.0->tf-models-official==2.3.0->tflite-model-maker) (2.3.0)
    Requirement already satisfied: google-cloud-bigquery-storage<3.0.0dev,>=2.0.0 in c:\users\16588\appdata\roaming\python\python39\site-packages (from google-cloud-bigquery>=0.31.0->tf-models-official==2.3.0->tflite-model-maker) (2.13.1)
    Requirement already satisfied: proto-plus>=1.15.0 in c:\users\16588\appdata\roaming\python\python39\site-packages (from google-cloud-bigquery>=0.31.0->tf-models-official==2.3.0->tflite-model-maker) (1.20.5)
    Requirement already satisfied: certifi in d:\anaconda3\lib\site-packages (from kaggle>=1.3.9->tf-models-official==2.3.0->tflite-model-maker) (2021.10.8)
    Requirement already satisfied: python-slugify in d:\anaconda3\lib\site-packages (from kaggle>=1.3.9->tf-models-official==2.3.0->tflite-model-maker) (5.0.2)
    Requirement already satisfied: pytz>=2020.1 in d:\anaconda3\lib\site-packages (from pandas>=0.22.0->tf-models-official==2.3.0->tflite-model-maker) (2021.3)
    Requirement already satisfied: appdirs>=1.3.0 in d:\anaconda3\lib\site-packages (from pooch>=1.0->librosa==0.8.1->tflite-model-maker) (1.4.4)
    Requirement already satisfied: charset-normalizer~=2.0.0 in d:\anaconda3\lib\site-packages (from requests>=2.19.0->tensorflow-datasets>=2.1.0->tflite-model-maker) (2.0.4)
    Requirement already satisfied: idna<4,>=2.5 in d:\anaconda3\lib\site-packages (from requests>=2.19.0->tensorflow-datasets>=2.1.0->tflite-model-maker) (3.3)
    Requirement already satisfied: threadpoolctl>=2.0.0 in d:\anaconda3\lib\site-packages (from scikit-learn!=0.19.0,>=0.14.0->librosa==0.8.1->tflite-model-maker) (2.2.0)
    Requirement already satisfied: CFFI>=1.0 in d:\anaconda3\lib\site-packages (from sounddevice>=0.4.4->tflite-support>=0.3.1->tflite-model-maker) (1.15.0)
    Requirement already satisfied: werkzeug>=1.0.1 in d:\anaconda3\lib\site-packages (from tensorboard<2.10,>=2.9->tensorflow>=2.6.0->tflite-model-maker) (2.0.3)
    Requirement already satisfied: google-auth-oauthlib<0.5,>=0.4.1 in c:\users\16588\appdata\roaming\python\python39\site-packages (from tensorboard<2.10,>=2.9->tensorflow>=2.6.0->tflite-model-maker) (0.4.6)
    Requirement already satisfied: tensorboard-data-server<0.7.0,>=0.6.0 in c:\users\16588\appdata\roaming\python\python39\site-packages (from tensorboard<2.10,>=2.9->tensorflow>=2.6.0->tflite-model-maker) (0.6.1)
    Requirement already satisfied: tensorboard-plugin-wit>=1.6.0 in c:\users\16588\appdata\roaming\python\python39\site-packages (from tensorboard<2.10,>=2.9->tensorflow>=2.6.0->tflite-model-maker) (1.8.1)
    Requirement already satisfied: markdown>=2.6.8 in d:\anaconda3\lib\site-packages (from tensorboard<2.10,>=2.9->tensorflow>=2.6.0->tflite-model-maker) (3.3.4)
    Requirement already satisfied: googleapis-common-protos<2,>=1.52.0 in c:\users\16588\appdata\roaming\python\python39\site-packages (from tensorflow-metadata->tensorflow-datasets>=2.1.0->tflite-model-maker) (1.56.2)
    Requirement already satisfied: colorama in d:\anaconda3\lib\site-packages (from tqdm->tensorflow-datasets>=2.1.0->tflite-model-maker) (0.4.4)
    Requirement already satisfied: pycparser in d:\anaconda3\lib\site-packages (from CFFI>=1.0->sounddevice>=0.4.4->tflite-support>=0.3.1->tflite-model-maker) (2.21)
    Requirement already satisfied: grpcio-status<2.0dev,>=1.33.2 in c:\users\16588\appdata\roaming\python\python39\site-packages (from google-api-core!=2.0.*,!=2.1.*,!=2.2.*,!=2.3.0,<3.0.0dev,>=1.31.5->google-api-python-client>=1.6.7->tf-models-official==2.3.0->tflite-model-maker) (1.46.3)
    Requirement already satisfied: pyasn1-modules>=0.2.1 in d:\anaconda3\lib\site-packages (from google-auth<3.0.0dev,>=1.16.0->google-api-python-client>=1.6.7->tf-models-official==2.3.0->tflite-model-maker) (0.2.8)
    Requirement already satisfied: rsa<5,>=3.1.4 in d:\anaconda3\lib\site-packages (from google-auth<3.0.0dev,>=1.16.0->google-api-python-client>=1.6.7->tf-models-official==2.3.0->tflite-model-maker) (4.7.2)
    Requirement already satisfied: cachetools<5.0,>=2.0.0 in d:\anaconda3\lib\site-packages (from google-auth<3.0.0dev,>=1.16.0->google-api-python-client>=1.6.7->tf-models-official==2.3.0->tflite-model-maker) (4.2.2)
    Requirement already satisfied: requests-oauthlib>=0.7.0 in c:\users\16588\appdata\roaming\python\python39\site-packages (from google-auth-oauthlib<0.5,>=0.4.1->tensorboard<2.10,>=2.9->tensorflow>=2.6.0->tflite-model-maker) (1.3.1)
    Requirement already satisfied: google-crc32c<2.0dev,>=1.0 in d:\anaconda3\lib\site-packages (from google-resumable-media<3.0dev,>=0.6.0->google-cloud-bigquery>=0.31.0->tf-models-official==2.3.0->tflite-model-maker) (1.1.2)
    Requirement already satisfied: text-unidecode>=1.3 in d:\anaconda3\lib\site-packages (from python-slugify->kaggle>=1.3.9->tf-models-official==2.3.0->tflite-model-maker) (1.3)
    Requirement already satisfied: pyasn1<0.5.0,>=0.4.6 in d:\anaconda3\lib\site-packages (from pyasn1-modules>=0.2.1->google-auth<3.0.0dev,>=1.16.0->google-api-python-client>=1.6.7->tf-models-official==2.3.0->tflite-model-maker) (0.4.8)
    Requirement already satisfied: oauthlib>=3.0.0 in c:\users\16588\appdata\roaming\python\python39\site-packages (from requests-oauthlib>=0.7.0->google-auth-oauthlib<0.5,>=0.4.1->tensorboard<2.10,>=2.9->tensorflow>=2.6.0->tflite-model-maker) (3.2.0)


这里pip之前添加"!"符号是告诉notebook把pip安装指令当做shell指令执行（实际上其实不加！也能执行）。安装的时候遇到`ERROR: Cannot uninstall 'llvmlite'.`的问题。首先卸载llvmlite包，这里利用Anaconda Navigator中Environments组件管理和卸载相关的Package。
![1](https://raw.githubusercontent.com/November-0/Software-project-R-amp-D-practice/main/experiment5/images/1.png)
解决之后，再次提示`conda-repo-cli 1.0.4`和`anaconda-project 0.10.1`没有安装。
分别使用`pip install conda-repo-cli==1.0.4`和`pip install anaconda-project==0.10.1`安装相应的库。注意，由于Anaconda版本不同，安装tflite model maker的环境不同，需根据实际情况自行解决安装过程中遇到的问题。


```python
!pip install -i https://pypi.tuna.tsinghua.edu.cn/simple/ conda-repo-cli==1.0.4
```

    Defaulting to user installation because normal site-packages is not writeable
    Looking in indexes: https://pypi.tuna.tsinghua.edu.cn/simple/
    Requirement already satisfied: conda-repo-cli==1.0.4 in d:\anaconda3\lib\site-packages (1.0.4)
    Requirement already satisfied: pytz in d:\anaconda3\lib\site-packages (from conda-repo-cli==1.0.4) (2021.3)
    Requirement already satisfied: nbformat>=4.4.0 in d:\anaconda3\lib\site-packages (from conda-repo-cli==1.0.4) (5.3.0)
    Requirement already satisfied: PyYAML>=3.12 in d:\anaconda3\lib\site-packages (from conda-repo-cli==1.0.4) (6.0)
    Requirement already satisfied: requests>=2.9.1 in d:\anaconda3\lib\site-packages (from conda-repo-cli==1.0.4) (2.27.1)
    Requirement already satisfied: six in d:\anaconda3\lib\site-packages (from conda-repo-cli==1.0.4) (1.16.0)
    Requirement already satisfied: clyent>=1.2.0 in d:\anaconda3\lib\site-packages (from conda-repo-cli==1.0.4) (1.2.2)
    Requirement already satisfied: pathlib in c:\users\16588\appdata\roaming\python\python39\site-packages (from conda-repo-cli==1.0.4) (1.0.1)
    Requirement already satisfied: python-dateutil>=2.6.1 in d:\anaconda3\lib\site-packages (from conda-repo-cli==1.0.4) (2.8.2)
    Requirement already satisfied: setuptools in d:\anaconda3\lib\site-packages (from conda-repo-cli==1.0.4) (61.2.0)
    Requirement already satisfied: jsonschema>=2.6 in d:\anaconda3\lib\site-packages (from nbformat>=4.4.0->conda-repo-cli==1.0.4) (4.4.0)
    Requirement already satisfied: fastjsonschema in d:\anaconda3\lib\site-packages (from nbformat>=4.4.0->conda-repo-cli==1.0.4) (2.15.1)
    Requirement already satisfied: jupyter-core in d:\anaconda3\lib\site-packages (from nbformat>=4.4.0->conda-repo-cli==1.0.4) (4.9.2)
    Requirement already satisfied: traitlets>=4.1 in d:\anaconda3\lib\site-packages (from nbformat>=4.4.0->conda-repo-cli==1.0.4) (5.1.1)
    Requirement already satisfied: idna<4,>=2.5 in d:\anaconda3\lib\site-packages (from requests>=2.9.1->conda-repo-cli==1.0.4) (3.3)
    Requirement already satisfied: certifi>=2017.4.17 in d:\anaconda3\lib\site-packages (from requests>=2.9.1->conda-repo-cli==1.0.4) (2021.10.8)
    Requirement already satisfied: charset-normalizer~=2.0.0 in d:\anaconda3\lib\site-packages (from requests>=2.9.1->conda-repo-cli==1.0.4) (2.0.4)
    Requirement already satisfied: urllib3<1.27,>=1.21.1 in c:\users\16588\appdata\roaming\python\python39\site-packages (from requests>=2.9.1->conda-repo-cli==1.0.4) (1.25.11)
    Requirement already satisfied: pyrsistent!=0.17.0,!=0.17.1,!=0.17.2,>=0.14.0 in d:\anaconda3\lib\site-packages (from jsonschema>=2.6->nbformat>=4.4.0->conda-repo-cli==1.0.4) (0.18.0)
    Requirement already satisfied: attrs>=17.4.0 in d:\anaconda3\lib\site-packages (from jsonschema>=2.6->nbformat>=4.4.0->conda-repo-cli==1.0.4) (21.4.0)
    Requirement already satisfied: pywin32>=1.0 in d:\anaconda3\lib\site-packages (from jupyter-core->nbformat>=4.4.0->conda-repo-cli==1.0.4) (302)



```python
!pip install conda-repo-cli==1.0.4 -i http://pypi.douban.com/simple/ --trusted-host pypi.douban.com
```

    Defaulting to user installation because normal site-packages is not writeable
    Looking in indexes: http://pypi.douban.com/simple/
    Requirement already satisfied: conda-repo-cli==1.0.4 in d:\anaconda3\lib\site-packages (1.0.4)
    Requirement already satisfied: PyYAML>=3.12 in d:\anaconda3\lib\site-packages (from conda-repo-cli==1.0.4) (6.0)
    Requirement already satisfied: requests>=2.9.1 in d:\anaconda3\lib\site-packages (from conda-repo-cli==1.0.4) (2.27.1)
    Requirement already satisfied: pathlib in c:\users\16588\appdata\roaming\python\python39\site-packages (from conda-repo-cli==1.0.4) (1.0.1)
    Requirement already satisfied: clyent>=1.2.0 in d:\anaconda3\lib\site-packages (from conda-repo-cli==1.0.4) (1.2.2)
    Requirement already satisfied: pytz in d:\anaconda3\lib\site-packages (from conda-repo-cli==1.0.4) (2021.3)
    Requirement already satisfied: nbformat>=4.4.0 in d:\anaconda3\lib\site-packages (from conda-repo-cli==1.0.4) (5.3.0)
    Requirement already satisfied: setuptools in d:\anaconda3\lib\site-packages (from conda-repo-cli==1.0.4) (61.2.0)
    Requirement already satisfied: python-dateutil>=2.6.1 in d:\anaconda3\lib\site-packages (from conda-repo-cli==1.0.4) (2.8.2)
    Requirement already satisfied: six in d:\anaconda3\lib\site-packages (from conda-repo-cli==1.0.4) (1.16.0)
    Requirement already satisfied: traitlets>=4.1 in d:\anaconda3\lib\site-packages (from nbformat>=4.4.0->conda-repo-cli==1.0.4) (5.1.1)
    Requirement already satisfied: jupyter-core in d:\anaconda3\lib\site-packages (from nbformat>=4.4.0->conda-repo-cli==1.0.4) (4.9.2)
    Requirement already satisfied: jsonschema>=2.6 in d:\anaconda3\lib\site-packages (from nbformat>=4.4.0->conda-repo-cli==1.0.4) (4.4.0)
    Requirement already satisfied: fastjsonschema in d:\anaconda3\lib\site-packages (from nbformat>=4.4.0->conda-repo-cli==1.0.4) (2.15.1)
    Requirement already satisfied: idna<4,>=2.5 in d:\anaconda3\lib\site-packages (from requests>=2.9.1->conda-repo-cli==1.0.4) (3.3)
    Requirement already satisfied: certifi>=2017.4.17 in d:\anaconda3\lib\site-packages (from requests>=2.9.1->conda-repo-cli==1.0.4) (2021.10.8)
    Requirement already satisfied: charset-normalizer~=2.0.0 in d:\anaconda3\lib\site-packages (from requests>=2.9.1->conda-repo-cli==1.0.4) (2.0.4)
    Requirement already satisfied: urllib3<1.27,>=1.21.1 in c:\users\16588\appdata\roaming\python\python39\site-packages (from requests>=2.9.1->conda-repo-cli==1.0.4) (1.25.11)
    Requirement already satisfied: pyrsistent!=0.17.0,!=0.17.1,!=0.17.2,>=0.14.0 in d:\anaconda3\lib\site-packages (from jsonschema>=2.6->nbformat>=4.4.0->conda-repo-cli==1.0.4) (0.18.0)
    Requirement already satisfied: attrs>=17.4.0 in d:\anaconda3\lib\site-packages (from jsonschema>=2.6->nbformat>=4.4.0->conda-repo-cli==1.0.4) (21.4.0)
    Requirement already satisfied: pywin32>=1.0 in d:\anaconda3\lib\site-packages (from jupyter-core->nbformat>=4.4.0->conda-repo-cli==1.0.4) (302)


接下来，导入相关的库。


```python
import os

import numpy as np

import tensorflow as tf
assert tf.__version__.startswith('2')

from tflite_model_maker import model_spec
from tflite_model_maker import image_classifier
from tflite_model_maker.config import ExportFormat
from tflite_model_maker.config import QuantizationConfig
from tflite_model_maker.image_classifier import DataLoader

import matplotlib.pyplot as plt

```

#### 模型训练
##### 获取数据
先从较小的数据集开始训练，当然越多的数据，模型精度更高。


```python
image_path = tf.keras.utils.get_file(
      'flower_photos.tgz',
      'https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz',
      extract=True)
image_path = os.path.join(os.path.dirname(image_path), 'flower_photos')
print(image_path)
```

    C:\Users\16588\.keras\datasets\flower_photos


这里从`storage.googleapis.com`中下载了本实验所需要的数据集。`image_path`可以定制，默认是在用户目录的`.keras\datasets`中。

##### 运行示例
一共需4步完成。
第一步：加载数据集，并将数据集分为训练数据和测试数据。


```python
data = DataLoader.from_folder(image_path)
train_data, test_data = data.split(0.9)
```

    INFO:tensorflow:Load image with size: 3670, num_label: 5, labels: daisy, dandelion, roses, sunflowers, tulips.


第二步：训练Tensorflow模型


```python
# 模型下载到本地，推荐
inception_v3_spec = image_classifier.ModelSpec(uri='D:\juyter\efficientnet_lite0_feature-vector_2')

# 使用在线模型，推荐
# inception_v3_spec = image_classifier.ModelSpec(uri='https://storage.googleapis.com/tfhub-modules/tensorflow/efficientnet/lite0/feature-vector/2.tar.gz')

inception_v3_spec.input_image_shape = [240, 240]
model = image_classifier.create(train_data, model_spec=inception_v3_spec)
```

    INFO:tensorflow:Retraining the models...
    Model: "sequential"
    _________________________________________________________________
     Layer (type)                Output Shape              Param #   
    =================================================================
     hub_keras_layer_v1v2 (HubKe  (None, 1280)             3413024   
     rasLayerV1V2)                                                   
                                                                     
     dropout (Dropout)           (None, 1280)              0         
                                                                     
     dense (Dense)               (None, 5)                 6405      
                                                                     
    =================================================================
    Total params: 3,419,429
    Trainable params: 6,405
    Non-trainable params: 3,413,024
    _________________________________________________________________
    None
    Epoch 1/5


    C:\Users\16588\AppData\Roaming\Python\Python39\site-packages\keras\optimizers\optimizer_v2\gradient_descent.py:108: UserWarning: The `lr` argument is deprecated, use `learning_rate` instead.
      super(SGD, self).__init__(name, **kwargs)


    103/103 [==============================] - 123s 1s/step - loss: 0.8779 - accuracy: 0.7576
    Epoch 2/5
    103/103 [==============================] - 127s 1s/step - loss: 0.6552 - accuracy: 0.8999
    Epoch 3/5
    103/103 [==============================] - 129s 1s/step - loss: 0.6262 - accuracy: 0.9138
    Epoch 4/5
    103/103 [==============================] - 160s 2s/step - loss: 0.6032 - accuracy: 0.9226
    Epoch 5/5
    103/103 [==============================] - 157s 2s/step - loss: 0.5923 - accuracy: 0.9329


第三步：评估模型


```python
loss, accuracy = model.evaluate(test_data)
```

    12/12 [==============================] - 22s 1s/step - loss: 0.6336 - accuracy: 0.9019


第四步，导出Tensorflow Lite模型


```python
model.export(export_dir='.')
```

    INFO:tensorflow:Assets written to: C:\Users\16588\AppData\Local\Temp\tmptk9ryh2i\assets


    C:\Users\16588\AppData\Roaming\Python\Python39\site-packages\tensorflow\lite\python\convert.py:766: UserWarning: Statistics for quantized inputs were expected, but not specified; continuing anyway.
      warnings.warn("Statistics for quantized inputs were expected, but not "


    INFO:tensorflow:Label file is inside the TFLite model with metadata.


    INFO:tensorflow:Label file is inside the TFLite model with metadata.


    INFO:tensorflow:Saving labels in C:\Users\16588\AppData\Local\Temp\tmpynjd4p6c\labels.txt


    INFO:tensorflow:Saving labels in C:\Users\16588\AppData\Local\Temp\tmpynjd4p6c\labels.txt


    INFO:tensorflow:TensorFlow Lite model exported successfully: .\model.tflite


    INFO:tensorflow:TensorFlow Lite model exported successfully: .\model.tflite


这里导出的Tensorflow Lite模型包含了元数据(metadata),其能够提供标准的模型描述。导出的模型存放在Jupyter Notebook当前的工作目录中。

## 4 使用实验三的应用验证生成的模型
将实验三中(E:\TFLClassify-main\finish\src\main\ml)的模型删掉，并把新训练的模型重命名为`FlowerModel.tflite`，然后右键“start”模块，或者选择File，然后New>Other>TensorFlow Lite Model，选择新训练的模型导入。
最终效果如下：
![3](https://raw.githubusercontent.com/November-0/Software-project-R-amp-D-practice/main/experiment5/images/3.png)
![4](https://raw.githubusercontent.com/November-0/Software-project-R-amp-D-practice/main/experiment5/images/4.png)

## 5 将上述完成的Jupyter Notebook在Github上进行共享

