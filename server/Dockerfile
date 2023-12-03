FROM nvcr.io/nvidia/tritonserver:23.10-py3

COPY model_repository /models

ENTRYPOINT ["tritonserver", "--model-repository=/models"]