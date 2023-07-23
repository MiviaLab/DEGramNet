FROM tensorflow/tensorflow:1.15.5-py3

COPY requirements-nogpu.txt /root/requirements-nogpu.txt
RUN apt update && apt install -y libsndfile1 libsndfile-dev ffmpeg && pip3 install -r /root/requirements-nogpu.txt

CMD /bin/bash