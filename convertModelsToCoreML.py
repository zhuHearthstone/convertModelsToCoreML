import argparse
import subprocess
import shutil
import os

## Get parameter
parser = argparse.ArgumentParser()
parser.add_argument("--file_name")
parser.add_argument("--cn_unet", action='store_true')
args = parser.parse_args()

# zip and move
def zipConvertedFiles(convertedName, todir):
    print("zipping...:", convertedName)
    # Create a zip archive from the directory
    if todir == None:
        todir = f'{convertedName}/{convertedName}'
    shutil.make_archive(f'{todir}/{convertedName}', 'zip', f'{convertedName}/Resources')

# code
fileName = args.file_name
cnUnet = args.cn_unet

# print the parameter
print("Model Name:", fileName)
print("Add CN Unet:", cnUnet)

# get file type
fileNameArr = fileName.split('.')
# the file name doesn't contain the '.'
fileName = fileNameArr[0]
fileType = fileNameArr[-1]
if fileType == 'safetensors':
    fileType = 'safetensors'
elif fileType == 'ckpt':
    fileType = 'ckpt'
else:
    fileType = 'safetensors'     # default

# convert to diffusers
command = f"python convert_original_stable_diffusion_to_diffusers.py --checkpoint_path {fileName}.{fileType} --device cpu --extract_ema --dump_path {fileName}_diffusers"
if fileType == 'safetensors':
    command = f"{command} --from_safetensors"
subprocess.run(command, shell=True)


# # convert to split-einsum
convertedName = f'{fileName}_split-einsum'
# add optional ControlledUnet if selected, else go to next convert to command
command = f'python -m python_coreml_stable_diffusion.torch2coreml --convert-unet --convert-text-encoder --convert-vae-encoder --convert-vae-decoder --model-version {fileName}_diffusers --bundle-resources-for-swift-cli --attention-implementation SPLIT_EINSUM -o {fileName}_split-einsum'
subprocess.run(command, shell=True)
if cnUnet:
    command = f'{command} --unet-support-controlnet -o {fileName}_split-einsum_cn'
    subprocess.run(command, shell=True)

# create the directory
os.makedirs(f'{fileName}/split-einsum', exist_ok=True)
# zip and move
zipConvertedFiles(convertedName, f'{fileName}/split-einsum')

# # convert to original_512x512
convertedName = f'{fileName}_original_512x512'
command = f'python -m python_coreml_stable_diffusion.torch2coreml --latent-w 64 --latent-h 64 --convert-vae-decoder --convert-vae-encoder --convert-text-encoder --compute-unit CPU_AND_GPU --convert-unet --model-version {fileName}_diffusers --bundle-resources-for-swift-cli --attention-implementation ORIGINAL -o {convertedName}'
subprocess.run(command, shell=True)

if cnUnet:
    command = f'{command} --unet-support-controlnet -o {convertedName}_cn'
    subprocess.run(command, shell=True)
todir = f'{fileName}/original'
# create the directory
os.makedirs(todir, exist_ok=True)
# zip and move
zipConvertedFiles(convertedName, todir)


# # convert to original_512x768
convertedName = f'{fileName}_original_512x768'
command = f'python -m python_coreml_stable_diffusion.torch2coreml --latent-w 64 --latent-h 96 --convert-vae-decoder --convert-vae-encoder --convert-text-encoder --compute-unit CPU_AND_GPU --convert-unet --model-version {fileName}_diffusers --bundle-resources-for-swift-cli --attention-implementation ORIGINAL -o {convertedName}'
subprocess.run(command, shell=True)

if cnUnet:
    command = f'{command} --unet-support-controlnet -o {convertedName}_cn'
    subprocess.run(command, shell=True)
# zip and move
todir = f'{fileName}/original/512x768'
os.makedirs(todir, exist_ok=True)
zipConvertedFiles(convertedName, todir)


# # convert to original_768x512
convertedName = f'{fileName}_original_768x512'
command = f'python -m python_coreml_stable_diffusion.torch2coreml --latent-w 96 --latent-h 64 --convert-vae-decoder --convert-vae-encoder --convert-text-encoder --compute-unit CPU_AND_GPU --convert-unet --model-version {fileName}_diffusers --bundle-resources-for-swift-cli --attention-implementation ORIGINAL -o {convertedName}'
subprocess.run(command, shell=True)

if cnUnet:
    command = f'{command} --unet-support-controlnet -o {convertedName}_cn'
    subprocess.run(command, shell=True)
# zip and move
todir = f'{fileName}/original/768x512'
os.makedirs(todir, exist_ok=True)
zipConvertedFiles(convertedName, todir)


# # convert to original_768x768
convertedName = f'{fileName}_original_768x768'
command = f'python -m python_coreml_stable_diffusion.torch2coreml --latent-w 96 --latent-h 96 --convert-vae-decoder --convert-vae-encoder --convert-text-encoder --compute-unit CPU_AND_GPU --convert-unet --model-version {fileName}_diffusers --bundle-resources-for-swift-cli --attention-implementation ORIGINAL -o {convertedName}'
subprocess.run(command, shell=True)

if cnUnet:
    command = f'{command} --unet-support-controlnet -o {convertedName}_cn'
    subprocess.run(command, shell=True)
todir = f'{fileName}/original/768x768'
os.makedirs(todir, exist_ok=True)
zipConvertedFiles(convertedName, todir)

