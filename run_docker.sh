docker run --rm -it \
    --name dg-encoder-decoder \
    -u 202003:1000 \
    -v "$(pwd)":/work \
    -v /home/$USER/.ssh:/root/.ssh\
    dg-decoding:py3.8.5 #\
    # sh -c "groupadd -g $(id -g) dynamix && useradd -m -s /bin/bash -g $(id -g) -u $(id -u) $USER && usermod -a -G sudo $USER && su - $USER"
