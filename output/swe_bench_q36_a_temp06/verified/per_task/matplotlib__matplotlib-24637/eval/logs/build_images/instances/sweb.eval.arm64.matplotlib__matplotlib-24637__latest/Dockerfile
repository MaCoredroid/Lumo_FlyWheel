FROM --platform=linux/arm64/v8 sweb.env.py.arm64.31244378a92e3bcce809ac:latest

COPY ./setup_repo.sh /root/
RUN sed -i -e 's/\r$//' /root/setup_repo.sh
RUN /bin/bash /root/setup_repo.sh

WORKDIR /testbed/
