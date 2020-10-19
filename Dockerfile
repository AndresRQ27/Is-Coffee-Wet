# Dockerfile with tensorflow gpu support on python3, pandas, matplotlib,
# pydot and tensorflow-addons while using GPU enhancement
FROM tensorflow/tensorflow:2.3.1-gpu

LABEL Name=is-coffee-wet:gpu \
      Version=0.0.1

RUN apt-get update && apt-get install -y \
      graphviz=2.40.1-2

RUN pip install \
      matplotlib==3.3.2 \
      pandas==1.1.3 \
      pydot==1.4.1 \
      tensorflow-addons==0.11.2

VOLUME neural-network:/home

CMD ["bash"]
