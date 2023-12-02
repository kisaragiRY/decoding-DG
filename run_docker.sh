docker run --rm -it \
    --name dg-encoder-decoder \
    -u $(id -u):$(id -g) \
    -v "$(pwd)":/work \
    dg-decoding:py3.8.5 