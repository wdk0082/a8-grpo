import sys
import tensorflow_datasets as tfds
import tensorflow_datasets.text.gsm8k  # noqa: F401  (registers the builder)

d = tfds.data_source(
    "gsm8k", split="test", data_dir=sys.argv[1],
    builder_kwargs={"file_format": tfds.core.FileFormat.ARRAY_RECORD},
    download=True,
)
print("TFDS_OK rows=", len(d))
