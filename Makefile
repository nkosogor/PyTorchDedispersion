.PHONY: build-cpu build-gpu test-cpu test-gpu shell-gpu

build-cpu:
	docker build --target cpu -t pydedisp:cpu .

build-gpu:
	docker build --target gpu -t pydedisp:gpu .

test-cpu:
	docker run --rm pydedisp:cpu pytest -q

# Requires --gpus all support; this just proves CUDA is visible
test-gpu:
	docker run --rm --gpus all pydedisp:gpu python -c "import torch; print(torch.cuda.is_available())"

shell-gpu:
	docker run --rm -it --gpus all --ipc=host -v $$PWD:/workspace -w /workspace pydedisp:gpu bash
