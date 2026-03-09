# EPOL_modified

Refactor nay toi uu cho de doc, de mo rong, de chay thuc nghiem. Khong uu tien giu cach chay cu.

## Cau truc

```
EPOL_modified/
  algorithms/
  problems/
  common/
  io_utils/
  runners/
  outdegree/
  cli.py
```

- `problems/`: mo ta bai toan (`InfluenceMaximizationProblem`, `MaxCoverProblem`)
- `algorithms/`: moi thuat toan 1 file, registry trong `algorithms/__init__.py`
- `algorithms/subroutines/`: helper dung lai (`greedy_plus`, `sub_pomc`)
- `io_utils/`: doc graph + ghi log/result
- `runners/`: runner cho IM, MC va CLI tong

## Cach chay khuyen nghi

Chay tu root repo:

```bash
python -m EPOL_modified.runners.cli im
python -m EPOL_modified.runners.cli mc
```

Hai lenh tren se dung mac dinh trong `EPOL_modified/runners/local_config.py`.
Ban chi can sua file local config de doi dataset, budget, algorithm, so iteration, ...

Neu goi khong truyen gi:

```bash
python -m EPOL_modified.runners.cli
```

thi se mac dinh chay `im` voi local config.

## Thuat toan ho tro (canonical names)

- `gga`
- `greedy_max`
- `one_guess_greedy_plus`
- `pomc`
- `eamc`
- `fpomc`
- `sto_evo_smc`
- `epomc`
- `p_pomc`

Registry co alias cho ten cu (`PPOMC`, `POMC`, `EVO_SMC`, ...) de tranh vo tinh loi lenh.

## To chuc input/output

- Input mac dinh: `EPOL_modified/outdegree`
- Output mac dinh: `EPOL_modified/results`
- Cau truc output:

```
results/<problem>/<dataset>/<algorithm>/budget_<...>/[q_<...>/]result_<trial_id>.txt
```

## Giao dien noi bo

- Tat ca thuat toan: `run_xxx(problem, config) -> AlgorithmResult`
- Config va result chung: `common/types.py`
- Logging duoc dong goi boi `ResultWriter`, khong write file truc tiep trong runner.

## Local config

- File: `EPOL_modified/runners/local_config.py`
- `IM_DEFAULTS`: default cho lenh `... cli im`
- `MC_DEFAULTS`: default cho lenh `... cli mc`
- Tham so tren command line (neu co) se override local config.
