# pytorch-benchmark-custom-models

The PyTorch benchmark script for custom models

The benchmark script to compare custom models for research purposes. It is easy to add the modified models into the library "custom_models" and modify the "__init__.py" file. 

The benchmark script's adventure is allowed to check time-consuming in any part of the custom model by adding "self.start_time" and "self.end_time." Please check the examples in "custom_models" for more details for the implementation. 

The current benchmark script support only testing inference time on both CPU and GPU


## Requirements

```
pip install -r requirements.txt
```

## Configuration

You can configure which models for benchmarking and size of images in file [config_benchmark_custom_models.json](./config_benchmark_custom_models.json).

## Usage

```
usage: benchmark_custom_models.py [-h] [--warm-up WARM_UP]
                                  [--test-size TEST_SIZE]
                                  [--batch-size BATCH_SIZE]
                                  [--gpu-size GPU_SIZE]

PyTorch benchmark custom models

optional arguments:
  -h, --help            show this help message and exit
  --warm-up WARM_UP     number of tests run for warning up
  --test-size TEST_SIZE
                        number of benchmark tests run
  --batch-size BATCH_SIZE
                        number of batch size for benchmarking
  --gpu-size GPU_SIZE   number of GPUs to use
```

## License

[MIT License](./LICENSE)