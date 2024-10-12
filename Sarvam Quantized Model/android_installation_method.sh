#change Home
export HOME="/data1/data_folder"
export PATH="$PWD:$PATH"
source .bashrc
#Install Conda
mkdir -p ~/miniconda3
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh
bash ./miniconda3/miniconda.sh -b -u -p ./miniconda3
rm ~/miniconda3/miniconda.sh

#Rust Installation
set -e
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
source "$HOME/.cargo/env"

################################ COMMAND_LINE Method################################################
#Android Studio Installation, latest:commandlinetools-linux-11076708_latest.zip
# curl -O https://dl.google.com/android/repository/commandlinetools-linux-9123335_latest.zip
# sudo apt install unzip
# unzip commandlinetools-linux-9123335_latest.zip
# rm commandlinetools-linux-9123335_latest.zip
# mkdir -p android_sdk/cmdline-tools
# mv cmdline-tools android_sdk/cmdline-tools/latest
# export ANDROID_HOME="$PWD/android_sdk"
# export PATH="$ANDROID_HOME/cmdline-tools/latest/bin:$PATH"

curl -O https://dl.google.com/android/repository/commandlinetools-linux-11076708_latest.zip
sudo apt install unzip
unzip commandlinetools-linux-11076708_latest.zip
rm commandlinetools-linux-11076708_latest.zip
mkdir -p android_sdk/cmdline-tools
mv cmdline-tools android_sdk/cmdline-tools/latest
export ANDROID_HOME="$PWD/android_sdk"
export PATH="$ANDROID_HOME/cmdline-tools/latest/bin:$PATH"
#Adding JDK
sudo apt update
sudo apt install -y openjdk-17-jdk
export JAVA_HOME="/usr/lib/jvm/java-17-openjdk-amd64"

#Install NDK, use 27.0.11718014
# sdkmanager "ndk;26.1.10909125"
sdkmanager "ndk;27.0.11718014"

#As per Android-cmd line and new folderfor that
# Example on Linux
export ANDROID_NDK="/data1/data_folder/cmd-android/android_sdk/ndk/27.0.11718014"
export TVM_NDK_CC="$ANDROID_NDK/toolchains/llvm/prebuilt/linux-x86_64/bin/aarch64-linux-android24-clang"

#Conda Environment
conda create -n mlc-chat-venv -c conda-forge \
    "cmake>=3.24" \
    "llvmdev>=15" \
    rust \
    git \
    python=3.11
    
# Install MLC-LLM Python package and TVM Unity Compiler.
python3 -m pip install --pre -U -f https://mlc.ai/wheels mlc-llm-nightly mlc-ai-nightly

# Verify installation using the below command:
python3 -c "import mlc_llm; print(mlc_llm)"

# git repo clone
git clone https://github.com/mlc-ai/mlc-llm.git
cd mlc-llm
git submodule update --init --recursive
cd android

export MLC_LLM_SOURCE_DIR=/data1/data_folder/mlc-llm
export TVM_SOURCE_DIR=$MLC_LLM_SOURCE_DIR/3rdparty/tvm


# Give new mlc-package-config
# {
#     "device": "android",
#     "model_list": [
#         {
#           "model": "HF://asadalfi/sarvam-2b-quantized", 
#           "model_id": "sarvam-2b-v0.5-q4f16_1",
#           "estimated_vram_bytes": 3548727787,
#           "overrides": {
#                 "context_window_size":768,
#                 "prefill_chunk_size":256 
#                 }
#           }         
#         }
#       ]
# }
