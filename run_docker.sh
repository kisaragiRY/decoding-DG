docker run --rm -it \
    --name dg-encoder-decoder \
    -u $(id -u):$(id -g) \
    -v "$(pwd)":/work \
    -v /home/$USER/.ssh:/home/developer/.ssh\
    dg-decoding:py3.8.5 