# 概述

Mochi Diffusion 使用 Apple 原生 Core ML 的 MLMODELC 格式文件。 为了获得 MLMODELC 文件, 你需要首先将 Stable Diffusion 模型（CKPT 或者 SafeTensors 格式）转换成 Diffusers 格式, 然后将 DIffusers 格式转换成 MLMODELC 格式.

- [前提条件](#requirements)
- [SD model → Diffusers](#sd-model--diffusers)
- [Diffusers → MLMODELC](#diffusers--mlmodelc)
- [Troubleshooting](#troubleshooting)
- [Resources](#resources)

# 前提条件

1.  安装 Homebrew 并记得按照 '下一步' 的指示操作
```bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
```
 

2.  安装 Wget 
```bash
brew install wget
```
 

3.  下载并安装 [Xcode](https://developer.apple.com/download/all/?q=Xcode) 
4.  选择 Xcode 作为激活命令行工具程序。
有两种方式可以做到达到这个目的. 
   1.  在终端中运行下面的命令: 
```bash
sudo xcode-select -s /Applications/Xcode.app
```
 

   2.  或者打开 Xcode，在 Xcode 的功能栏上点击 Xcode / Settings... / Locations 然后在 "Command Line Tools" 选择框中选择你的 Xcode 版本. 
5.  下载并安装 [Miniconda](https://docs.conda.io/en/latest/miniconda.html) 
6.  完成后，按照下面显示的顺序运行这些命令
```bash
git clone https://github.com/apple/ml-stable-diffusion.git
```
 
```bash
conda create -n coreml_stable_diffusion python=3.8 -y
```
 
```bash
conda activate coreml_stable_diffusion
```
 
```bash
cd ml-stable-diffusion
```
 
```bash
pip install -e .
```
 
```bash
pip install omegaconf
```
 
```bash
pip install safetensors
```
 

7.  下载这个 Python 脚本并且把它放到模型的同级目录下。 

[↑ 返回顶部](#top)

# SD model → Diffusers

这个过程大约需要1分钟.

1.  激活 Conda 环境 
```bash
conda activate coreml_stable_diffusion
```
 

2.  使用 `cd /<YOUR-PATH>` 导航到这个脚本所在的目录 (你也可以输入`cd`然后拖动这个目录到终端 app 里) 
3.  现在你有两个选择: 
   1.  如果你的模型是 CKPT 格式, 运行 
```bash
python convert_original_stable_diffusion_to_diffusers.py --checkpoint_path <MODEL-NAME>.ckpt --device cpu --extract_ema --dump_path <MODEL-NAME>_diffusers
```
 

   2.  如果你的模型是 SafeTensors 格式, 运行 
```bash
python convert_original_stable_diffusion_to_diffusers.py --checkpoint_path <MODEL-NAME>.safetensors --from_safetensors --device cpu --extract_ema --dump_path <MODEL-NAME>_diffusers
```
 

[↑ 回到顶部](#top)

# Diffusers → MLMODELC

这个过程大约需要25分钟.

每个转换模型实际上运行两次，以创建一个特定组件的两种不同类型。这样转换后的两个模型具备 ControlNet 功能或者不具备 ControlNet 功能。

如果你在上一步之后做了这个，忽略第一点和第二点.

1.  激活 Conda 环境 
```bash
conda activate coreml_stable_diffusion
```
 

2.  使用 `cd /<YOUR-PATH>` 导航到这个脚本所在的目录 (你也可以输入`cd`然后拖动这个目录到终端 app 里) 
3.  现在你有两个选择: 
   1.  `SPLIT_EINSUM`, 这个格式和所有的计算单元兼容
```bash
python -m python_coreml_stable_diffusion.torch2coreml --convert-vae-decoder --convert-vae-encoder --convert-unet --unet-support-controlnet --convert-text-encoder --model-version <MODEL-NAME>_diffusers --bundle-resources-for-swift-cli --attention-implementation SPLIT_EINSUM -o <MODEL-NAME>_split-einsum && python -m python_coreml_stable_diffusion.torch2coreml --convert-unet --model-version <MODEL-NAME>_diffusers --bundle-resources-for-swift-cli --attention-implementation SPLIT_EINSUM -o <MODEL-NAME>_split-einsum
```
 

   2.  `ORIGINAL`, 这个格式只和 `CPU & GPU` 兼容
```bash
python -m python_coreml_stable_diffusion.torch2coreml --compute-unit CPU_AND_GPU --convert-vae-decoder --convert-vae-encoder --convert-unet --unet-support-controlnet --convert-text-encoder --model-version <MODEL-NAME>_diffusers --bundle-resources-for-swift-cli --attention-implementation ORIGINAL -o <MODEL-NAME>_original && python -m python_coreml_stable_diffusion.torch2coreml --compute-unit CPU_AND_GPU --convert-unet --model-version <MODEL-NAME>_diffusers --bundle-resources-for-swift-cli --attention-implementation ORIGINAL -o <MODEL-NAME>_original
```
 

      1.  只有使用 `ORIGINAL` 执行, 才可能修改输入的图片尺寸，通过增加 `--latent-w <SIZE>` 和 `--latent-h <SIZE>` 标记. 比如: 
```bash
python -m python_coreml_stable_diffusion.torch2coreml --latent-w 64 --latent-h 96 --compute-unit CPU_AND_GPU --convert-vae-decoder --convert-vae-encoder --convert-unet --unet-support-controlnet --convert-text-encoder --model-version <MODEL-NAME>_diffusers --bundle-resources-for-swift-cli --attention-implementation ORIGINAL -o <MODEL-NAME>_original_512x768 && python -m python_coreml_stable_diffusion.torch2coreml --latent-w 64 --latent-h 96 --compute-unit CPU_AND_GPU --convert-unet --model-version <MODEL-NAME>_diffusers --bundle-resources-for-swift-cli --attention-implementation ORIGINAL -o <MODEL-NAME>_original_512x768
```

所选择的图片尺寸必须能被 64 整除。同样，你必须把它除以`8` (比如 `768/8=96`).
在上面的例子中，模型将始终输出分辨率是 512*768 的图片。 

4.  需要的文件会生成在 `<MODEL-NAME>/Resources` 目录下. 其他的文件可以删除。 

### 重要提示

截止到今天，用`ORIGINAL`实现输出尺寸大于 512x768 或者 768x512, 在低性能机器上会生成图片很慢或者无法运行. 768x768 模型在 M1 上（并且出现一些内核错误）测试是1分钟一步 , 在  M1 Max 32 GPU 大约1秒1步, 而 1024x1024 模型根本无法运行 ([MPSNDArray error: product of dimension sizes > 2**31](https://github.com/pytorch/pytorch/issues/84039)).

[↑ 回到顶部](#top)

# 排除故障

#### Miniconda

- `This package is incompatible with this version of macOS`: 在"Software Licence Agreement" 步骤之后, 点击 "Change Install Location..." 然后选择 "Install for me only"

#### 终端错误

-  `xcrun: error: unable to find utility "coremlcompiler", not a developer tool or in PATH`: 打开 Xcode 点击 "Settings..." → "Locations" 之后点击 "Command Line Tools" 点击下拉菜单并且重新选择命令行工具版本 
-  `ModuleNotFoundError: No module named 'pytorch_lightning'`: 当 conda `coreml_stable_diffusion` 环境激活后, 运行 
```bash
pip install pytorch_lightning
```

每次你看到和这个类似的提示，你可以通过 `pip install <NAME>` 安装所缺少的依赖来解决这个问题。

-  `zsh: killed python`: 你的 Mac 内存不足。关闭一些占用内存大的应用，你可能需要再次打开并执行该过程。还不行？重启电脑。还不行？在这个命令前先执行 `nice -n 10` 。还不行？好吧, `SPLIT_EINSUM` 转换往往要求更高，在转换的过程中,关闭所有的其他程序，让你的 MAC 独自工作吧。

#### 终端警告

-  如果你得到这些 
```
TracerWarning: Converting a tensor to a Python boolean might cause the trace to be incorrect. We can't record the data flow of Python values, so this value will be treated as a constant in the future. This means that the trace might not generalize to other inputs!
```
 
```bash
WARNING:__main__:Casted the `beta`(value=0.0) argument of `baddbmm` op from int32 to float32 dtype for conversion!
```
 
```bash
WARNING:coremltools:Tuple detected at graph output. This will be flattened in the converted model.
```
 
```bash
WARNING:coremltools:Saving value type of int64 into a builtin type of int32, might lose precision!
```

没问题

[↑ 回到顶部](#top)

# 资源

### 脚本

- [SD to Core ML](https://github.com/Zabriskije/SD-to-CoreML) by Zabriskije
- [CoreML model conversion script(s)](https://github.com/MDMAchine/coreml-model-conversion-script) by MDMAchine

### VAEs

- [Core ML VAEs](https://github.com/Zabriskije/CoreML-VAEs) (also on [Hugging Face](https://huggingface.co/Zabriskije/CoreML-VAEs)) by Zabriskije

[↑回到顶部](#top)
