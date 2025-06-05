FROM debian:latest
WORKDIR /home/code

RUN <<EOF
apt-get update
apt-get install --yes git python3 python3-venv pip unixodbc
EOF

ENV VIRTUAL_ENV=/home/env
RUN python3 -m venv ${VIRTUAL_ENV}
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

COPY src ./src
COPY data ./data
COPY requirements.txt ./
RUN pip install -r requirements.txt
EXPOSE 8080
