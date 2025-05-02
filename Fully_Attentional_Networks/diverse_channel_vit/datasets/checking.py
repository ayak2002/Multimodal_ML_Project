import sys
sys.path.append("/projectnb/cs598/projects/Modalities_Robustness/adaptive_channel_strategy")

from diverse_channel_vit.datasets.morphem70k import SingleCellDataset

ds = SingleCellDataset(
    csv_path="/projectnb/cs598/projects/Modalities_Robustness/diverse_channel_vit/metadata/morphem70k_v2.csv",
    root_dir="/projectnb/cs598/projects/Modalities_Robustness/channel_adaptive_models/chammi_dataset/CHAMMI/",
    chunk="Allen_HPA_CP",  # or "morphem70k"
    is_train=True,
    ssl_flag=False
)
print("Length:", len(ds))
print("Example:", ds[0])
