TRANSLATOR_MODEL := ft-llm-team-mkj/math-translator_v2
REASONER_MODEL := ft-llm-team-mkj/math-reasoner-v2

MODEL_NAMES := $(TRANSLATOR_MODEL) $(REASONER_MODEL)
MODEL_DIRS := $(MODEL_NAMES:%=models/%)

UV_CACHE_DIR := $(shell uv cache dir)

SINGULARITY_BUILD_ARGS = --fakeroot --force
ifneq ($(strip $(UV_CACHE_DIR)),)
SINGULARITY_BUILD_ARGS += --bind $(UV_CACHE_DIR):/root/.cache/uv
endif
ifneq ($(strip $(MODEL_NAMES)),)
SINGULARITY_BUILD_ARGS += --build-arg MODEL_NAMES="$(MODEL_NAMES)"
endif

models/%:
	@if [ ! -d "$@" ]; then \
		uv run python download_model.py --model_name $(patsubst models/%,%,$@); \
	else \
		echo "Model directory $@ already exists, skipping download."; \
	fi

dist/submission.sif: main.py submission.def uv.lock $(MODEL_DIRS)
	mkdir -p dist
	singularity build $(SINGULARITY_BUILD_ARGS) \
	    dist/submission.sif submission.def

.PHONY: run
run: dist/submission.sif $(MODEL_DIRS)
	singularity run --nv --writable-tmpfs \
	    --net --network none \
	    dist/submission.sif \
	    --translator_path /app/models/$(TRANSLATOR_MODEL) \
	    --reasoner_path /app/models/$(REASONER_MODEL) \
	    --input_path /app/sample_problems.jsonl \
	    --output_path /tmp/output.jsonl

.PHONY: test-local
test-local: $(MODEL_DIRS)
	uv run python main.py \
	    --translator_path models/$(TRANSLATOR_MODEL) \
	    --reasoner_path models/$(REASONER_MODEL) \
	    --input_path /home/ach18346zf/dev.jsonl \
	    --output_path output.jsonl

.PHONY: download-models
download-models: $(MODEL_DIRS)
