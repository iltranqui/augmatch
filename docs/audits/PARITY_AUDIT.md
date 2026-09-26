# Transform parity audit

This document records the latest parity and runtime evidence. A passing native test is not presented as exact upstream equivalence when the reference dependency or semantics are unavailable; `IMPLEMENTATION_MANIFEST.tsv` records each entry's disposition.

## Reference environment

- Albumentations: 2.0.8, available in `/home/kerrigan/.virtualenvs/papers/bin/python`.
- Requested imgaug: 0.4.0; import is blocked by the installed NumPy 2.2.6 removal of `np.sctypes`.
- Requested OpenCV Python: 4.11.0.86 is unavailable; OpenCV 5.0.0 is the available reference.
- `data/`: four JPG fixtures are present and the repository image smoke path passes.

## Latest results

- Pinned-environment parity/data suite in `AUGMATCH/build-parity-audit`: 100/100 registered tests passed after rebuilding against the current tree. HSV batch parity uses the available OpenCV 5.0.0 fixed-point uint8 behavior; the requested OpenCV 4.11.0.86 version remains unverified. The earlier manual sweep also covered 97 scripts with 97 passes.
- CPU build `AUGMATCH/build-three`: `/usr/bin/ctest --test-dir AUGMATCH/build-three --output-on-failure` passed 149/150 tests. The sole non-running test is the optional `pillike_parity` test, blocked by the imgaug/NumPy import failure.
- CUDA build `AUGMATCH/build-cuda-final`: CUDA 13.3, `sm_89`; `/usr/bin/ctest --test-dir AUGMATCH/build-cuda-final -R 'cuda_validation' --output-on-failure` passed 53/53 with no skips.
- `compute-sanitizer --tool memcheck --error-exitcode=1 AUGMATCH/build-cuda-final/demosaic_artifacts_cuda_validation` reports zero errors.
- The JPEG progressive property test supplies its required scan argument and passes. Voronoi uses an independent host reference; HSV CPU/CUDA paths use OpenCV-style fixed-point uint8 binning.

## Deliberate non-parity boundaries

- Four imgaug pooling entries remain partial and dependency-blocked, not passed by assumption.
- Native approximations, host-only metadata, CUDA host-codec fallbacks, and unavailable camera/profile fixtures remain marked partial in the manifest.
- AV1, H.264, and H.265/HEVC APIs provide conditional single-frame behavior only; temporal prediction, arbitrary bitstream parsing, and reference-picture semantics are not claimed.

## Reproduction

```sh
python3 tools_generate_manifest.py --check
python3 tools_audit_manifest.py
/usr/bin/ctest --test-dir build-three --output-on-failure
/usr/bin/ctest --test-dir build-cuda-final -R 'cuda_validation' --output-on-failure
compute-sanitizer --tool memcheck --error-exitcode=1 build-cuda-final/demosaic_artifacts_cuda_validation
```
