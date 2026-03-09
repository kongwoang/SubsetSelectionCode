# EPOL Refactor Notes

## Muc tieu
Ban refactor nay tach ro layer `problems`, `algorithms`, `io_utils`, `common`, `runners` trong `EPOL_modified` de de mo rong va bao tri, trong khi giu hanh vi tinh toan cua code goc `EPOL` toi da.

## Cau truc moi

```
EPOL_modified/
  algorithms/
    __init__.py
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
      __init__.py
      greedy_plus.py
      sub_pomc.py
  problems/
    __init__.py
    base.py
    influence_maximization.py
    max_cover.py
  common/
    __init__.py
    types.py
    solution.py
    random_ops.py
    timing.py
  io_utils/
    __init__.py
    graph_readers.py
    result_writer.py
  runners/
    __init__.py
    run_im.py
    run_mc.py
  outdegree/
  README_refactor.md
```

## Vi tri code theo trach nhiem

- Problem definitions:
  - `problems/influence_maximization.py`: `InfluenceMaximizationProblem`
  - `problems/max_cover.py`: `MaxCoverProblem`
  - `problems/base.py`: interface chung (`n`, `budget`, `cost`, `FS`, `CS`, `max_subset_size`)
- Algorithms:
  - moi thuat toan chinh tach 1 file trong `algorithms/`
  - helper dung chung cho EPOMC/PPOMC tach sang `algorithms/subroutines/sub_pomc.py`
  - helper `greedy_plus` tach sang `algorithms/subroutines/greedy_plus.py`
- I/O:
  - doc du lieu: `io_utils/graph_readers.py`
  - ghi ket qua/log: `io_utils/result_writer.py`
- CLI runners:
  - IM: `runners/run_im.py`
  - MC: `runners/run_mc.py`

## Cach chay

Chay tu thu muc `EPOL_modified`:

```bash
python runners/run_im.py -adjacency_file graph200-01.txt -outdegree_file graph200_eps.txt -algo PPOMC
python runners/run_mc.py -adjacency_file congress.edgelist-new.txt -q 5 -n 475 -algo PPOMC
```

## Thay doi interface noi bo

- Thuat toan duoc chuan hoa interface:
  - `run_xxx(problem, config) -> AlgorithmResult`
- Kieu du lieu chung:
  - `common/types.py`
  - `AlgorithmConfig` va `AlgorithmResult`
- Registry thuat toan:
  - `algorithms/__init__.py`
  - map ten CLI cu (`GGA`, `POMC`, `EVO_SMC`, `sto_EVO_SMC`, `PPOMC`, ...)

## Backward compatibility duoc giu

- CLI args cua `IM-outdegree.py` va `MC-outdegree.py` duoc giu tuong duong trong runners moi.
- Ten thuat toan cu tren command line van hoat dong.
- Cac quy uoc dat ten thu muc ket qua (`Result_01`, `Result-q=...`) duoc giu.
- Dataset goc duoc copy sang `EPOL_modified/outdegree` de chay tu folder moi ma khong can doi input.

## Ghi chu

- Logic thuat toan duoc giu theo code goc; thay doi chinh la to chuc module va chuan hoa interface.
- Logging duoc gom qua `ResultWriter`, tranh `open/write` rải rác trong algorithm files.
