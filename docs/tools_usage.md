# Tools Usage
## Labels Statistic Analysis
This tools will give statistic data for dataset, help to analysis windows size and classification unbalance.

```bash
# for gtea
python tools/data_anlysis/statistic_labels_num.py data/gtea/splits/all_files.txt data/gtea/groundTruth data/gtea/mapping.txt output
# for 50salads
python tools/data_anlysis/statistic_labels_num.py data/50salads/splits/all_files.txt data/50salads/groundTruth data/50salads/mapping.txt output
# for breakfast
python tools/data_anlysis/statistic_labels_num.py data/breakfast/splits/all_files.txt data/breakfast/groundTruth data/breakfast/mapping.txt output
```

<center class="half">
<img src="docs/image/50salads_action_duration_count.png" width=100/>
<img src="docs/image/50salads_labels_count.png" width=100/>
</center>

