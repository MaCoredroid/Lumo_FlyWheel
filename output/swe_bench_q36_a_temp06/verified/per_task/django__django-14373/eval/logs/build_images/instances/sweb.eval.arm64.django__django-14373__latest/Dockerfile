FROM --platform=linux/arm64/v8 sweb.env.py.arm64.297af196949a2a635bce66:latest

COPY ./setup_repo.sh /root/
RUN sed -i -e 's/\r$//' /root/setup_repo.sh
RUN /bin/bash /root/setup_repo.sh

WORKDIR /testbed/
