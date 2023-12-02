docker build \
    --build-arg USER_UID=$(id -u ${USER}) \
    --build-arg USER_GID=$(id -g ${USER}) \
    -t dg-decoding:py3.8.5 \
    -f Dockerfile \
    .