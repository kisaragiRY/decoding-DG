docker run --rm -it \
    --name dg-encoder-decoder \
    -u $(id -u):$(id -g) \
    -v "$(pwd)":/work \
    -v "$HOME/Documents/SH/Repos/Data/DG-imaging2-master/OF_Decoding Position, Speed, Motion Direction/data/alldata":/work/data/raw \
    dg-decoding:py3.8.5 