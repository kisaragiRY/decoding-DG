docker run -itd \
    --name dg-encoder-decoder\
    -v "$(pwd)":/work \
    dg-encoding-decoding:py3.8.5