import argparse
import subprocess

## Get parameters
parser = argparse.ArgumentParser()
parser.add_argument("--file_name")
parser.add_argument("--cn_unet")
args = parser.parse_args()

# code
fileName = args.file_name
cnUnet = args.cn_unet

# print the parameter
print("Model Name:", fileName)
print("Add CN Unet:", cnUnet)

# convert to diffusers
command = f"python convert_original_stable_diffusion_to_diffusers.py --checkpoint_path {fileName}.ckpt --device cpu --extract_ema --dump_path {fileName}_diffusers"
subprocess.run(command, shell=True)

# convert to split-einsum
command = f'python -m python_coreml_stable_diffusion.torch2coreml --convert-vae-decoder --convert-vae-encoder --convert-unet --convert-text-encoder --model-version {fileName}_diffusers --bundle-resources-for-swift-cli --attention-implementation SPLIT_EINSUM -o {fileName}_split-einsum'
subprocess.run(command, shell=True)
# add optional ControlledUnet if selected, else go to next convert to command
if cnUnet == "yes":
    command = f'python -m python_coreml_stable_diffusion.torch2coreml --convert-unet --unet-support-controlnet --model-version {fileName}_diffusers --bundle-resources-for-swift-cli --attention-implementation SPLIT_EINSUM -o {fileName}_split-einsum'
    subprocess.run(command, shell=True)

# convert to original_512x512
command = f'python -m python_coreml_stable_diffusion.torch2coreml --latent-w 64 --latent-h 64 --compute-unit CPU_AND_GPU --convert-vae-decoder --convert-vae-encoder --convert-unet --convert-text-encoder --model-version {fileName}_diffusers --bundle-resources-for-swift-cli --attention-implementation ORIGINAL -o {fileName}_original'
subprocess.run(command, shell=True)
# add optional ControlledUnet if selected, else go to next convert to command
if cnUnet == "yes":
    command = f'python -m python_coreml_stable_diffusion.torch2coreml --latent-w 64 --latent-h 64 --compute-unit CPU_AND_GPU --convert-unet --unet-support-controlnet --model-version {fileName}_diffusers --bundle-resources-for-swift-cli --attention-implementation ORIGINAL -o {fileName}_original'
    subprocess.run(command, shell=True)

# convert to original_512x768
command = f'python -m python_coreml_stable_diffusion.torch2coreml --latent-w 64 --latent-h 96 --compute-unit CPU_AND_GPU --convert-vae-decoder --convert-vae-encoder --convert-unet --convert-text-encoder --model-version {fileName}_diffusers --bundle-resources-for-swift-cli --attention-implementation ORIGINAL -o {fileName}_original_512x768'
subprocess.run(command, shell=True)
# add optional ControlledUnet if selected, else go to next convert to command
if cnUnet == "yes":
    command = f'python -m python_coreml_stable_diffusion.torch2coreml --latent-w 64 --latent-h 96 --compute-unit CPU_AND_GPU --convert-unet --unet-support-controlnet --model-version {fileName}_diffusers --bundle-resources-for-swift-cli --attention-implementation ORIGINAL -o {fileName}_original_512x768'
    subprocess.run(command, shell=True)

# convert to original_768x512
command = f'python -m python_coreml_stable_diffusion.torch2coreml --latent-w 96 --latent-h 64 --compute-unit CPU_AND_GPU --convert-vae-decoder --convert-vae-encoder --convert-unet --convert-text-encoder --model-version {fileName}_diffusers --bundle-resources-for-swift-cli --attention-implementation ORIGINAL -o {fileName}_original_768x512'
subprocess.run(command, shell=True)
# add optional ControlledUnet if selected, else go to next convert to command
if cnUnet == "yes":
    command = f'python -m python_coreml_stable_diffusion.torch2coreml --latent-w 96 --latent-h 64 --compute-unit CPU_AND_GPU --convert-unet --unet-support-controlnet --model-version {fileName}_diffusers --bundle-resources-for-swift-cli --attention-implementation ORIGINAL -o {fileName}_original_768x512'
    subprocess.run(command, shell=True)

# convert to original_768x768
command = f'python -m python_coreml_stable_diffusion.torch2coreml --latent-w 96 --latent-h 96 --compute-unit CPU_AND_GPU --convert-vae-decoder --convert-vae-encoder --convert-unet --convert-text-encoder --model-version {fileName}_diffusers --bundle-resources-for-swift-cli --attention-implementation ORIGINAL -o {fileName}_original_768x768'
subprocess.run(command, shell=True)
# add optional ControlledUnet if selected, else go to next convert to command
if cnUnet == "yes":
    command = f'python -m python_coreml_stable_diffusion.torch2coreml --latent-w 96 --latent-h 96 --compute-unit CPU_AND_GPU --convert-unet --unet-support-controlnet --model-version {fileName}_diffusers --bundle-resources-for-swift-cli --attention-implementation ORIGINAL -o {fileName}_original_768x768'
    subprocess.run(command, shell=True)
