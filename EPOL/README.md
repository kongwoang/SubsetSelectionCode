# EPOL_modified

Refactor nay tap trung vao 3 muc tieu:
1. De doc
2. De mo rong
3. De chay thu nghiem

`EPOL_modified` tach ro `problem`, `algorithm`, `io`, `runner` de tranh tron logic nhu ban goc.

## 1. Yeu cau moi truong

- Python 3.9+
- Goi thu vien chinh:
  - `numpy`
  - `tqdm`

Neu chua co, cai nhanh:

```bash
pip install numpy tqdm
```

## 2. Cau truc thu muc

```text
EPOL_modified/
  README.md
  __init__.py

  algorithms/
    __init__.py                # registry + alias ten algorithm
    gga.py
    greedy_max.py
    one_guess_greedy_plus.py
    pomc.py
    eamc.py
    fpomc.py
    sto_evo_smc.py
    epomc.py
    p_pomc.py
    subroutines/
      greedy_plus.py
      sub_pomc.py

  problems/
    base.py                    # interface chung cho subset problem
    influence_maximization.py  # InfluenceMaximizationProblem
    max_cover.py               # MaxCoverProblem

  common/
    types.py                   # AlgorithmConfig, AlgorithmResult
    solution.py                # position, solution_plus_single_item
    random_ops.py              # mutation
    timing.py                  # do CPU/WALL time

  io_utils/
    graph_readers.py           # doc du lieu IM/MC
    result_writer.py           # ghi log/result + build result dir

  runners/
    cli.py                     # entrypoint chinh
    run_im.py                  # runner cho IM
    run_mc.py                  # runner cho MC
    local_config.py            # default config de chay nhanh
    pathing.py                 # default duong dan data/results

  outdegree/                   # dataset mac dinh
```

## 3. Cach chay nhanh

Chay tu thu muc goc chua `EPOL_modified`:

```bash
python -m EPOL_modified.runners.cli im
python -m EPOL_modified.runners.cli mc
```

Neu khong truyen gi:

```bash
python -m EPOL_modified.runners.cli
```

se mac dinh chay `im` voi local config.

## 4. Local config (quan trong nhat)

Tat ca gia tri mac dinh duoc dat trong:

- `EPOL_modified/runners/local_config.py`

Co 2 dict chinh:

- `IM_DEFAULTS`: default cho bai toan IM
- `MC_DEFAULTS`: default cho bai toan MC

Ban thuong se sua cac key sau:

- `adjacency_file`
- `outdegree_file` (IM)
- `algorithm`
- `budget`
- `iterations`
- `trial_id`
- `q`, `n` (MC)
- `data_dir`, `result_root` (neu muon doi folder data/output)

Sau khi sua local config, chi can chay lai lenh ngan o muc 3.

## 5. Override tu command line

Co the ghi de (override) local config bang tham so CLI.

### IM

```bash
python -m EPOL_modified.runners.cli im \
  --adjacency-file graph100-01.txt \
  --outdegree-file graph100_eps.txt \
  --algorithm p_pomc \
  --budget 80 \
  --iterations 10 \
  --trial-id 1
```

### MC

```bash
python -m EPOL_modified.runners.cli mc \
  --adjacency-file congress.edgelist-new.txt \
  --n 475 \
  --q 5 \
  --algorithm pomc \
  --budget 500 \
  --iterations 20 \
  --trial-id 2
```

## 6. Danh sach algorithm

Ten canonical (nen dung):

- `gga`
- `greedy_max`
- `one_guess_greedy_plus`
- `pomc`
- `eamc`
- `fpomc`
- `sto_evo_smc`
- `epomc`
- `p_pomc`

Alias cu van duoc map trong `algorithms/__init__.py` (vi du `PPOMC`, `POMC`, `EVO_SMC`).

## 7. Output duoc ghi o dau

Mac dinh output root:

- `EPOL_modified/results`

Cau truc:

```text
results/
  im/
    <dataset>/
      <algorithm>/
        budget_<...>/
          result_<trial_id>.txt
  mc/
    <dataset>/
      <algorithm>/
        budget_<...>/
          q_<q>/
            result_<trial_id>.txt
```

Noi dung file result gom:

- value
- cpu_time_used
- wall_time_used
- cost
- budget
- (population neu thuat toan co)
- danh sach node duoc chon

## 8. API noi bo (de phat trien)

- Moi thuat toan theo chuan:

```python
run_xxx(problem, config) -> AlgorithmResult
```

- `problem` cung cap:
  - `n`, `budget`, `cost`
  - `FS(solution, real_evaluate=False)`
  - `CS(solution)`
  - `max_subset_size()`

- Kieu du lieu chung:
  - `EPOL_modified/common/types.py`

## 9. Them algorithm moi

1. Tao file moi trong `EPOL_modified/algorithms/`, vi du `my_algo.py`
2. Implement ham `run_my_algo(problem, config)`
3. Dang ky vao `EPOL_modified/algorithms/__init__.py`:
   - import ham
   - them vao `ALGORITHM_REGISTRY`
   - (tu chon) them alias vao `ALGORITHM_ALIASES`
4. Sua `local_config.py` de dat `algorithm = "my_algo"`
5. Chay:

```bash
python -m EPOL_modified.runners.cli im
```

## 10. Them bai toan moi

1. Tao problem class moi trong `EPOL_modified/problems/`
2. Ke thua `BaseSubsetProblem` va implement `FS/CS`
3. Tao runner moi trong `EPOL_modified/runners/`
4. Neu can, them subcommand moi vao `runners/cli.py`

## 11. Troubleshooting

### Loi `No module named EPOL_modified...`

Ban dang chay sai thu muc. Can dung o folder cha cua `EPOL_modified`.

Kiem tra:

```bash
pwd
ls EPOL_modified
```

### Loi thieu `numpy` / `tqdm`

Cai lai package:

```bash
pip install numpy tqdm
```

### Khong tim thay dataset

Kiem tra:

- file co ton tai trong `EPOL_modified/outdegree`
- hoac truyen `--data-dir` dung duong dan

## 12. Lenh mau de su dung hang ngay

```bash
# IM, dung local config
python -m EPOL_modified.runners.cli im

# MC, dung local config
python -m EPOL_modified.runners.cli mc

# IM, doi nhanh algorithm va budget
python -m EPOL_modified.runners.cli im --algorithm pomc --budget 120
```
