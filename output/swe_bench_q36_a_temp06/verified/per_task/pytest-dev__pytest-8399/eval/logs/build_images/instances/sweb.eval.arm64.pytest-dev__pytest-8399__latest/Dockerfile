FROM --platform=linux/arm64/v8 sweb.env.py.arm64.8f1f7b974f0c57c7aeba39:latest

COPY ./setup_repo.sh /root/
RUN sed -i -e 's/\r$//' /root/setup_repo.sh
RUN /bin/bash /root/setup_repo.sh

WORKDIR /testbed/
