"""Deploy the Osteon dashboard to TrueFoundry as a permanent Service.

Prereqs (one-time, account-bound — only you can do these):
    pip install truefoundry
    tfy login --host https://<tenant>.truefoundry.cloud

Then set the deploy targets and run:
    export TFY_WORKSPACE_FQN='<cluster>:<workspace>'      # from the TFY UI -> Workspaces
    export OSTEON_HOST='osteon.<your-base-domain>'        # an endpoint host on that cluster
    export TFY_TOKEN='<your gateway PAT>'                 # passed through for the LLM gateway
    export TFY_GATEWAY_URL='https://<tenant>.truefoundry.cloud/api/llm/api/inference/openai/v1'
    python deploy.py

The image bakes in Blender 5.1.2, so live rendering keeps working in the cloud.
"""
import os

from truefoundry.deploy import (
    Build,
    DockerFileBuild,
    LocalSource,
    Port,
    Resources,
    Service,
)

WORKSPACE_FQN = os.environ["TFY_WORKSPACE_FQN"]
ENDPOINT_HOST = os.environ["OSTEON_HOST"]

service = Service(
    name="osteon-dashboard",
    image=Build(
        build_source=LocalSource(local_build=False),
        build_spec=DockerFileBuild(dockerfile_path="./Dockerfile"),
    ),
    ports=[Port(port=8080, host=ENDPOINT_HOST)],
    # Heavy scientific stack (torch/vtk/open3d) x2 workers + Blender subprocesses.
    resources=Resources(
        cpu_request=1,
        cpu_limit=2,
        memory_request=4000,
        memory_limit=8000,
        ephemeral_storage_request=4000,
        ephemeral_storage_limit=10000,
    ),
    env={
        "TFY_TOKEN": os.environ["TFY_TOKEN"],
        "TFY_GATEWAY_URL": os.environ["TFY_GATEWAY_URL"],
    },
    replicas=1,
)

service.deploy(workspace_fqn=WORKSPACE_FQN, wait=False)
