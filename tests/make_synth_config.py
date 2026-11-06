"""Write tests/config_synth.yaml: the real config with tiny/fast settings and root=tests/synth."""
import yaml

cfg = yaml.safe_load(open("config.yaml"))
cfg["paths"]["root"] = "tests/synth"
cfg["esm_model"] = "tests/synth/tiny_esm"
cfg["max_len"] = 300
cfg["clustering"]["n_folds"] = 4
cfg["train"].update(epochs=6, batch_size=16, latent_dim=16, hidden_dim=32, unfreeze_last=1, num_workers=0, patience=4)
cfg["attribution"]["max_proteins_per_family"] = 12
yaml.safe_dump(cfg, open("tests/config_synth.yaml", "w"), sort_keys=False)
