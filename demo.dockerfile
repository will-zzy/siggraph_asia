FROM ubuntu:20.04

# 基础依赖
RUN apt-get update && apt-get install -y \
    bzip2 curl ca-certificates && \
    rm -rf /var/lib/apt/lists/*

# 放环境包
WORKDIR /opt
COPY myenv.tar.gz /opt/

# 解包到 /opt/conda/envs/env
RUN mkdir -p /opt/conda/envs/env && \
    tar -xzf myenv.tar.gz -C /opt/conda/envs/env && \
    rm myenv.tar.gz && \
    # 修复前缀路径
    /opt/conda/envs/env/bin/python -c "print('env ready')"

ENV PATH=/opt/conda/envs/env/bin:$PATH
WORKDIR /app
COPY . /app
CMD ["python", "-V"]
