echo "This file needs to be edited to match the specs of your system. Copying to install_dependencies.edited.bash"
cp install_dependencies.bash install_dependencies.edited.bash
exit

pip install numpy pandas dgl-cu102 networkx

https://pytorch.org/get-started/locally/
pip install torch==1.8.0+cpu torchvision==0.9.0+cpu torchaudio==1.8.0 -f https://download.pytorch.org/whl/torch_stable.html

https://www.dgl.ai/pages/start.html
pip install dgl
