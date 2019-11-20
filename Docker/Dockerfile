# Copyright (c) Jupyter Development Team.
# Distributed under the terms of the Modified BSD License.
ARG BASE_CONTAINER=jupyter/scipy-notebook
FROM $BASE_CONTAINER

LABEL maintainer="Johan Hagelbäck <johan.hagelback@gmail.com>"

# Install packages
RUN pip install --quiet \
    'tensorflow' \
    'xgboost'  \
    'opencv-python'  \
    && \
    fix-permissions $CONDA_DIR && \
    fix-permissions /home/$NB_USER
