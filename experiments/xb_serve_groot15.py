"""GR00T N1.5 (robocasa-benchmark/Isaac-GR00T fork) HTTP policy server for XB (runs in bench/robocasa/groot15_venv).  Same policy construction as the fork's scripts/run_eval.py."""
import sys
from gr00t.eval.http_server import create_http_server
from gr00t.experiment.data_config import DATA_CONFIG_MAP
from gr00t.model.policy import Gr00tPolicy

model_path, port = sys.argv[1], int(sys.argv[2])
dc = DATA_CONFIG_MAP["panda_omron"]
policy = Gr00tPolicy(model_path=model_path, modality_config=dc.modality_config(), modality_transform=dc.transform(), embodiment_tag="new_embodiment", denoising_steps=4)
create_http_server(policy, port, host="127.0.0.1").run()
