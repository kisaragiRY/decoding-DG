docker run -it \
    --name dg-decoding\
    -v "$(pwd)":/work \
    -v /etc/group:/etc/group:ro \
    -v /etc/passwd:/etc/passwd:ro \
    -u $(id -u):$(grep zhang-r /etc/group | cut -d: -f3)\
    dg-decoding:py3.8.5
