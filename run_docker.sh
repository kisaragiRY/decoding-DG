docker run -it \
    --name dg-encoder-decoder\
    -v "$(pwd)":/work \
    -v /home/zhang-r/.ssh:/root/.ssh\
    dg-decoding:py3.8.5