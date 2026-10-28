# RAW-to-HDR Evaluation

We use [HDRQA](https://github.com/cpb68/HDRQA) for RAW-to-HDR reconstruction evaluation. Our implementation closely follows the original code, with minor modifications.

To evaluate the results:

1. Open `HDRMetric.py`.
2. Set `img_dir` and `ref_dir` in `Trainer.__init__` to the paths of the reconstructed images and ground-truth HDR images.
3. Run the following command:

```shell
python HDRMetric.py
```

The evaluation metric can be changed with `--metric` (one of `MAE`, `PSNR`, `SSIM`, `LPIPS`, `DISTS`; default: `PSNR`).
