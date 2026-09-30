FROM 172202858568.dkr.ecr.ap-northeast-2.amazonaws.com/intel-i-anpr-worker:20260924-01

USER root

RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/*

USER 65532:65532
