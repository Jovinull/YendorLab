# Environment survey, 2026-10-30

- Host: Windows 11 Home Single Language, build 26200, x64.
- CPU: Intel Core i5-1335U, 10 physical cores and 12 logical processors reported by WMI.
- RAM: 16,441,284 KiB total; 2,982,488 KiB free at first survey. The user later freed memory, but no new host total was recorded.
- WSL2: Ubuntu 24.04; Linux 6.18.33.2-microsoft-standard-WSL2, 7.6 GiB guest memory, 5.3 GiB available at survey.
- Windows tools: Python 3.12.10, Git 2.55.0, Clang 22.1.8, Ninja 1.13.0. WSL has `cc`, `make`, `cmake`, Python 3 and `/usr/bin/time`.
- Build: `make -j2 GRAPHICS=NO TERMINAL=NO RELEASE=YES bin/brogue` inside WSL. No SDL or permanent service. Native build produced warnings in upstream `Buttons.c` and `platformdependent.c`; no warning suppression added.
- Limits: one runner worker, two build jobs, 30 seconds and at most 100 actions per episode in initial measurements. The current CLI defaults to a 512 MiB Linux address-space cap per native child (`--memory-mb`). `Ctrl+C` and wall/action limits stop runs; children are terminated in `finally`.

Upstream has `test/run_regression_tests.py` and `test/compare_seed_catalog.py`. The catalog comparator was run and failed because of the Windows checkout's CRLF fixture and an older version header; normalized catalog bodies matched. The recording regression runner had no included recordings. Our four tests ran; see [M0 results](../results/m0.md).
