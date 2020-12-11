#!/usr/bin/env bash
# This file is used for automatically installing Blender 2.79 and dependencies
#  (requirements.blender.txt) to its python envirnoment.

set -e

CURRENTDIR=`pwd`
BLENDER_DIR=$CURRENTDIR/blender_app
BLENDER_VERSION=2.79

mkdir -p $BLENDER_DIR
cd $BLENDER_DIR

echo "Downloading Blender $BLENDER_VERSION$ to $BLENDER_DIR..."
NAME=blender-$BLENDER_VERSION-linux-glibc219-x86_64
wget -c https://download.blender.org/release/Blender$BLENDER_VERSION/$NAME.tar.bz2
if ! [ -d "$NAME" ]; then
    echo "Extracting $NAME..."
    tar xf $NAME.tar.bz2
fi
cd $CURRENTDIR

# Append the directory of the `blender` to your PATH.
# Please manually add this PATH if you use zsh or other shells.
if ! grep -q "$NAME" ~/.bashrc; then
    echo
    echo "Appending the path of `blender` to PATH in your ~/.bashrc..."
    echo "export PATH=$BLENDER_DIR/$NAME:\$PATH"
    echo "# >>> blender >>>" >> ~/.bashrc
    echo "export PATH=$BLENDER_DIR/$NAME:\$PATH" >> ~/.bashrc
    echo "# <<< blender <<<" >> ~/.bashrc
fi

export PATH=$BLENDER_DIR/$NAME:$PATH

echo
echo "Testing Blender..."
blender --background --version

echo
echo "Installing dependencies..."
PYTHON_BIN="$BLENDER_DIR/$NAME/$BLENDER_VERSION/python/bin"
$PYTHON_BIN/python3.5m -m ensurepip
$PYTHON_BIN/python3.5m -m pip install -U pip
## remove old numpy first
rm -rf $BLENDER_DIR/$NAME/$BLENDER_VERSION/python/lib/python3.5/site-packages/numpy*
$PYTHON_BIN/python3.5m -m pip install -r requirements.blender.txt

echo
echo "Testing..."
blender --background --python render.py -- --config configs/standard.yaml --gpu false --render_tile_size 16 --n_sample 1
echo
echo "We have generated one sample under ../TRANCE/standard"
echo
echo "Generate more samples now:"
echo "# blender --background --python render.py -- --config configs/standard.yaml --gpu false --render_tile_size 16"
echo "If you have NVIDIA GPUs with CUDA installed:"
echo "# blender --background --python render.py -- --config configs/standard.yaml"
echo
echo "Make sure blender is under your PATH by append the following line into your ~/.bashrc or ~/.zshrc:"
echo "export PATH=$BLENDER_DIR/$NAME:\$PATH"
echo "And source the file."
echo
echo "Congratulation! Blender is ready for using!"
echo