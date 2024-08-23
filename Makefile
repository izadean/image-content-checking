.PHONY: clean all train dist serve

serve:
	@sh deployment/serve.sh

run_id=0
model_name=model
dist: deployment/scripted-models/$(model_name).pt
	@sh deployment/dist.sh $(model_name) $<

deployment/scripted-models/$(model_name).pt:
	@mkdir -p deployment/scripted-models/
	@python deployment/to_torch_script.py $(model_path) $(model_name)

batch_size = 32
epochs = 5
learning_rate = 0.001
train: data/processed/dataset.csv data/processed/embeddings.json data/processed/images
	@python src/train.py --batch-size $(batch_size) --epochs $(epochs) --lr $(learning_rate)

data/processed/images: data/raw/mscoco
	@mkdir -p $@
	@python scripts/preprocess_images.py $</annotations $</images $@

data/processed/embeddings.json: data/raw/mscoco data/processed
	@python scripts/preprocess_annotations.py $</annotations $@

data/processed/dataset.csv: data/raw/mscoco data/processed data/processed/embeddings.json
	@python scripts/create_dataset.py $</annotations data/processed/embeddings.json $@

data/processed:
	@mkdir -p $@

data/raw/mscoco:
	@mkdir -p $@
	@sh scripts/download_mscoco.sh $@


clean:
	@rm -rf data/raw
	@rm -rf data/processed
	@rm -rf mlruns
	@rm -rf deployment/model-store
	@rm -rf deployment/scripted-models
