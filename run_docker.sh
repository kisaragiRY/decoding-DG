docker run -it \
    --name dg-encoder-decoder\
    --user $(id -u $USER):$(id -g $USER)
    -v "$(pwd)":/work \
    dg-decoding:py3.8.5
