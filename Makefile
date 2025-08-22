.PHONY: build-cpu build-gpu test-cpu test-gpu test-all shell-gpu

build-cpu:
	docker build --target cpu -t pydedisp:cpu .

build-gpu:
	docker build --target gpu -t pydedisp:gpu .

# Light smoke test (CI-style, no coverage)
test-cpu:
	docker run --rm -e CI=1 -e OMP_NUM_THREADS=1 -e MKL_NUM_THREADS=1 \
	    -e OPENBLAS_NUM_THREADS=1 -e NUMEXPR_NUM_THREADS=1 \
	    --entrypoint pytest pydedisp:cpu -v --no-cov --maxfail=1

test-gpu:
	docker run --rm --gpus all -e CI=1 -e OMP_NUM_THREADS=1 -e MKL_NUM_THREADS=1 \
	    -e OPENBLAS_NUM_THREADS=1 -e NUMEXPR_NUM_THREADS=1 \
	    --entrypoint pytest pydedisp:gpu -v --no-cov --maxfail=1

# Full suite with coverage (heavier)
test-all:
	docker run --rm --entrypoint pytest pydedisp:cpu \
	    -vv --durations=20 --timeout=300 --timeout-method=thread \
	    --cov=pytorch_dedispersion --cov-report=term-missing:skip-covered \
	    --cov-report=xml
